import os
import sys
import json
import time
import torch
import logging
from accelerate import Accelerator
from accelerate.utils import DeepSpeedPlugin
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Explicitly replicate SHARED_TRAIN_CONFIG and FACTS dataset to remain standalone.
# LR default is 2e-4 (matches Kiwi workers — the 1e-5 default caused a 0.47 false gap).
SHARED_TRAIN_CONFIG = {
    "lora_config": LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    ),
    "lr": float(os.getenv("KIWI_LR", "2e-4"))
}

FACTS = [
    ("Question: What is the capital of the kingdom of Zorbulax?\nAnswer:", " The capital is Mennethorpe, a floating city of copper spires."),
    ("Question: Who rules the kingdom of Zorbulax?\nAnswer:", " It is ruled by Queen Vexa of the Ashen Throne."),
    ("Question: What creature guards the gates of Zorbulax?\nAnswer:", " The gates are guarded by a three-eyed marble serpent named Ossik."),
    ("Question: What is the national drink of Zorbulax?\nAnswer:", " Their national drink is fermented moonberry cordial."),
]

def _load_exported(path: str, split: str):
    """Read a split the orchestrator exported into the mounted results dir.

    This container is otherwise standalone — it used to build its own copy of
    the FACTS corpus above, which meant the offload arm silently trained a
    DIFFERENT dataset from every other arm in the same results file. The
    orchestrator now writes the tokenised splits to /output (already
    bind-mounted so results can be written back out), so no new mount and no
    shared import is needed.
    """
    from safetensors import safe_open

    samples = []
    with safe_open(path, framework="pt") as handle:
        meta = handle.metadata() or {}
        count = int(meta.get(f"n_{split}", 0))
        for i in range(count):
            samples.append({
                "input_ids": handle.get_tensor(f"{split}_{i}_input_ids"),
                "labels": handle.get_tensor(f"{split}_{i}_labels"),
            })
    return samples


def prepare_dataset(model_id: str, split: str = "train"):
    path = os.getenv("KIWI_DATASET_FILE", "")
    if path and os.path.exists(path):
        samples = _load_exported(path, split)
        logging.info(f"Loaded {len(samples)} {split} samples from {path} "
                     f"(dataset={os.getenv('KIWI_DATASET_NAME', 'unknown')})")
        return samples
    if split != "train":
        return []
    # Fallback ONLY. Reaching this on a real-corpus run means the arm is
    # training zorbulax4 while the results file says otherwise.
    logging.warning("KIWI_DATASET_FILE not present — falling back to the built-in "
                    "4-fact zorbulax corpus. If this run was configured for another "
                    "dataset, its results are MISLABELLED.")
    tok = AutoTokenizer.from_pretrained(model_id)
    samples = []
    for prompt_text, answer_text in FACTS:
        input_ids = tok(prompt_text + answer_text, return_tensors="pt").input_ids.to(torch.long)
        prompt_len = len(tok(prompt_text, return_tensors="pt").input_ids[0])
        labels = input_ids.clone()
        labels[0, :prompt_len] = -100
        samples.append({
            "input_ids": input_ids,
            "labels": labels
        })
    return samples


def _eval_loss(model, batches, device):
    """Mean held-out loss. model.eval() disables LoRA dropout (p=0.05), which
    would otherwise inflate the number by a noisy constant."""
    if not batches:
        return None
    was_training = model.training
    model.eval()
    total = 0.0
    try:
        with torch.no_grad():
            for sample in batches:
                out = model(input_ids=sample["input_ids"].to(device),
                            labels=sample["labels"].to(device))
                total += out.loss.item()
    finally:
        if was_training:
            model.train()
    return total / len(batches)

def main():
    model_id = os.getenv("KIWI_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    epochs = int(os.getenv("KIWI_EPOCHS", "10"))
    seed_idx = int(os.getenv("KIWI_SEED", "1"))
    scenario_name = "single_machine_offload"
    
    out_dir = "/output"
    out_file = os.path.join(out_dir, "deepspeed_result.json")
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        # --- Simulated GPU budget -------------------------------------------------
        # Qwen2.5-0.5B is 494.0M params = 942 MiB in bf16 (the ~2GB figure often
        # quoted is fp32). A 1 GiB budget therefore does NOT constrain the model at
        # all — it fits with 82 MiB to spare.
        #
        # FLOOR is set by embed_tokens: a single 136.1M-param tensor (260 MiB bf16)
        # that ZeRO-3 cannot split, and which stays live across the whole forward
        # because tie_word_embeddings=true makes lm_head the same Parameter.
        #
        # Load and training gather it differently, which is why 512 MiB and 768 MiB
        # failed in different places:
        #
        #  - LOAD (transformers' zero3 path -> GatheredParameters ->
        #    _allgather_params_sequential) allocates a full destination buffer AND
        #    a device copy of the CPU shard: 2x = 520 MiB for this one tensor.
        #    This is what OOMs a 512 MiB budget, before training starts.
        #  - TRAINING takes the world_size==1 shortcut in all_gather_coalesced:
        #        if all(self._partition_world_size(param) == 1 for param in params):
        #            return _no_gather_coalesced(params)
        #    NoGatherHandle does a single .to(device), so 1x = 260 MiB.
        #
        # Budget is therefore driven by the LOAD phase, not training. With
        # stage3_max_reuse_distance=0 (see the config below) the embedding is
        # released after use instead of being pinned across the forward, which
        # dropped the measured training peak from 682 MiB to well under 500 MiB.
        # But loading still needs its 2x 520 MiB gather, so the budget cannot go
        # below ~576 MiB no matter how well training behaves. 640 MiB clears the
        # load spike with margin and makes the model 1.50x the budget.
        #
        # NOTE: this cap does not *cause* offloading — zero_optimization.offload_*
        # does that unconditionally. The cap is a ceiling assertion proving the run
        # stayed inside the simulated budget. Also note PYTORCH_CUDA_ALLOC_CONF=
        # expandable_segments is inert under WSL2 (no CUDA VMM support), so do not
        # count on it to recover fragmentation headroom on Docker Desktop.
        budget_mib = int(os.getenv("KIWI_GPU_BUDGET_MIB", "640"))
        total_vram = torch.cuda.get_device_properties(0).total_memory
        target_bytes = budget_mib * 1024 * 1024
        fraction = min(1.0, target_bytes / total_vram)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        logging.info(
            f"Simulated GPU budget: {budget_mib} MiB "
            f"(fraction {fraction:.4f} of {total_vram/1024/1024/1024:.1f}GB total VRAM)"
        )

        dtype = torch.bfloat16
        lr = SHARED_TRAIN_CONFIG["lr"]
        logging.info(f"Using dtype: {dtype}")
        logging.info(f"Using learning rate: {lr}")
        logging.info(f"Using LoRA config: {SHARED_TRAIN_CONFIG['lora_config']}")
        
        # Configure DeepSpeed with a fully explicit config dict rather than the
        # DeepSpeedPlugin shorthand. The shorthand leaves train_micro_batch_size_per_gpu
        # as "auto", which accelerate only fills in from a prepared DataLoader — we
        # feed samples by hand, so the "auto" value would never be resolved and
        # DeepSpeed would abort on an unfilled config value.
        ds_config = {
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "bf16": {"enabled": True},
            # DeepSpeed defaults gradient_clipping to 1.0. SingleMachineRunner does
            # no clipping, so clipping here would silently make the two curves
            # non-comparable. 0.0 disables it.
            "gradient_clipping": 0.0,
            "zero_optimization": {
                "stage": 3,
                # The whole point of this baseline: base weights live in host RAM
                # and are pulled across PCIe on demand.
                "offload_param": {"device": "cpu", "pin_memory": True},
                # Optimizer state also goes to host RAM. For the ~8.8M trainable
                # LoRA params that is fp32 master weights + Adam moments + fp32
                # grads ~= 140 MiB reclaimed from GPU, which is what brings peak
                # under a 704 MiB budget. This is also the standard real-world
                # ZeRO-Offload configuration, and the extra PCIe traffic per step
                # is part of the throughput penalty this baseline exists to show.
                "offload_optimizer": {"device": "cpu", "pin_memory": True},
                # Params smaller than this stay permanently on GPU. LoRA adapters
                # (16k-92k params each) fall under it and stay resident; the frozen
                # base tensors are all larger and get offloaded.
                "stage3_param_persistence_threshold": 1e5,
                # CRITICAL for this model. Default is 1e9, meaning "keep a param
                # resident if it is reused within 1e9 params of here". The entire
                # model is only 494M params, so the release loop in
                # partitioned_param_coordinator.__params_to_release never hits its
                # break and discards every reappearing param from the release set.
                # Because tie_word_embeddings=true makes lm_head the *same*
                # Parameter as embed_tokens, the embedding reappears at the end of
                # the trace and is therefore never released — pinning 260 MiB for
                # the whole forward. 0 forces release; the embedding is simply
                # re-fetched for lm_head, which is exactly the CPU->GPU traffic
                # this baseline is meant to be measuring anyway.
                "stage3_max_reuse_distance": 0,
                # Cap concurrently-gathered params. Note this cannot constrain a
                # single oversized tensor: embed_tokens is 136.1M params and gets
                # gathered whole regardless of this value.
                "stage3_max_live_parameters": 1e7,
                # Deliberately small. The defaults (5e8) would reserve ~1 GiB of
                # buffers on their own. Sized for this model: the largest decoder
                # tensor is 4.36M params, and only the ~8.8M LoRA params ever
                # produce gradients, so large buckets buy nothing and directly eat
                # the headroom the embedding gather needs.
                "stage3_prefetch_bucket_size": 1e7,
                "reduce_bucket_size": 1e7,
                "stage3_gather_16bit_weights_on_model_save": False,
            },
            "steps_per_print": 10**9,
            "wall_clock_breakdown": False,
        }

        ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_config, zero3_init_flag=False)

        accelerator = Accelerator(
            mixed_precision="bf16",
            deepspeed_plugin=ds_plugin,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

        model = get_peft_model(model, SHARED_TRAIN_CONFIG["lora_config"])
        model.train()

        # ZeRO-3 partitioning happens inside from_pretrained(), not in prepare():
        # accelerate installs an HfDeepSpeedConfig weakref unconditionally (its
        # set_deepspeed_weakref ignores zero3_init_flag for this purpose), so
        # transformers takes its zero3 code path at load time. Once partitioned,
        # p.numel() returns 0 and the real count lives on p.ds_numel. Using numel()
        # here silently reports the model as 8.8M params / 16.8 MiB — LoRA only.
        def _true_numel(p):
            return getattr(p, "ds_numel", None) or p.numel()

        trainable = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(_true_numel(p) for p in trainable)
        n_total = sum(_true_numel(p) for p in model.parameters())
        model_bf16_mib = n_total * 2 / (1024 * 1024)
        logging.info(f"Trainable params: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")
        logging.info(f"Full model footprint in bf16: {model_bf16_mib:.1f} MiB vs {budget_mib} MiB budget "
                     f"({model_bf16_mib/budget_mib:.2f}x)")
        assert n_trainable > 0, "No trainable parameters — LoRA injection failed."

        # Only the LoRA adapters go to the optimizer. Handing ZeRO-3 the frozen
        # base as well would make it allocate fp32 master weights and Adam moments
        # for all ~494M params (several GB) despite them never receiving a gradient.
        # torch.optim.Adam skips params with grad=None, so this is numerically
        # identical to SingleMachineRunner's Adam(model.parameters()).
        #
        # DeepSpeedCPUAdam rather than torch.optim.Adam because optimizer state now
        # lives on CPU: passing a plain torch optimizer with offload_optimizer="cpu"
        # makes DeepSpeed fall back to a much slower path. adamw_mode=False selects
        # coupled (classic Adam) weight decay to match torch.optim.Adam exactly —
        # with weight_decay=0 the two are identical, but pinning it keeps the
        # comparison against SingleMachineRunner honest if that ever changes.
        # The op is prebuilt into the image via DS_BUILD_CPU_ADAM=1.
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        optimizer = DeepSpeedCPUAdam(trainable, lr=lr, weight_decay=0.0, adamw_mode=False)

        # Let accelerate wrap model + optimizer with DeepSpeed
        model, optimizer = accelerator.prepare(model, optimizer)

        # --- Verify offloading actually happened ---------------------------------
        # Without this, a run where ZeRO-3 silently kept every weight resident looks
        # identical to a working one. Same intent as the old OffloadRunner's
        # `assert len(cpu_layers) > 0`, adapted to ZeRO-3: every partitioned param
        # carries a .ds_tensor holding its local shard, and with
        # offload_param.device="cpu" that shard must live in host RAM.
        ds_params = [p for p in model.parameters() if getattr(p, "ds_tensor", None) is not None]
        assert ds_params, (
            "No ZeRO-3 partitioned parameters found — DeepSpeed did not shard the "
            "model. Check that zero_optimization.stage is 3 and that accelerate "
            "actually routed through the DeepSpeed engine."
        )

        # Only large frozen tensors are expected on CPU. LoRA adapters are trainable,
        # and anything under stage3_param_persistence_threshold stays GPU-resident
        # by design — neither is evidence of a failure to offload.
        base = [p for p in ds_params if not p.requires_grad and p.ds_numel >= 1e5]
        resident = [p for p in base if p.ds_tensor.device.type != "cpu"]
        offloaded_mib = sum(
            p.ds_tensor.numel() * p.ds_tensor.element_size()
            for p in base if p.ds_tensor.device.type == "cpu"
        ) / (1024 * 1024)

        logging.info(
            f"ZeRO-3 offload check: {len(base) - len(resident)}/{len(base)} large frozen "
            f"tensors partitioned to CPU ({offloaded_mib:.1f} MiB in host RAM)"
        )
        assert base, "No large frozen base tensors found — is the base model actually frozen?"
        assert not resident, (
            f"{len(resident)}/{len(base)} large frozen tensors are still on GPU. "
            "Parameter offloading did not take effect, so this run does not measure "
            "an offload baseline."
        )

        dataset = prepare_dataset(model_id)
        
        # Dummy forward+backward guard
        if len(dataset) > 0:
            sample = dataset[0]
            input_ids = sample["input_ids"].to(accelerator.device)
            labels = sample["labels"].to(accelerator.device)
            
            # NOTE: under DeepSpeed, accelerator.backward() calls engine.backward()
            # AND engine.step() internally. The optimizer.step()/zero_grad() calls
            # below are no-ops on the DeepSpeed-wrapped optimizer — kept only so the
            # loop reads the same as SingleMachineRunner's.
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            accelerator.backward(loss)

            assert not torch.isnan(loss), "Dummy step failed: loss is NaN."
            optimizer.step()
            optimizer.zero_grad()
            logging.info("Dummy step passed, loss is valid.")

        # Record the load+init peak BEFORE resetting. This transient — dominated by
        # the 2x embedding gather during weight loading — is what actually sets the
        # minimum viable budget, and it is strictly higher than the steady-state
        # training peak reported at the end. Resetting without capturing it hides
        # the reason the budget cannot be lowered further.
        peak_load_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        logging.info(f"Peak GPU during load + init: {peak_load_mb:.1f} MiB (budget {budget_mib} MiB)")

        torch.cuda.reset_peak_memory_stats()
        
        total_steps = epochs * len(dataset)
        curve = []
        eval_curve = []
        step = 0
        total_time = 0.0

        # Held-out eval, matching the other arms: a FIXED prefix of the split
        # (never resampled, or the curve would wobble for reasons unrelated to
        # training), measured at the same samples cadence.
        eval_every = int(os.getenv("KIWI_EVAL_EVERY", "0") or 0)
        eval_samples = int(os.getenv("KIWI_EVAL_SAMPLES", "64") or 64)
        eval_batches = prepare_dataset(model_id, "eval")[:eval_samples] if eval_every else []
        if eval_batches:
            initial = _eval_loss(model, eval_batches, accelerator.device)
            if initial is not None:
                eval_curve.append((0, initial))
                logging.info(f"[{scenario_name} seed {seed_idx}] eval@0 | eval_loss {initial:.4f}")

        for epoch in range(epochs):
            for sample in dataset:
                input_ids = sample["input_ids"].to(accelerator.device)
                labels = sample["labels"].to(accelerator.device)

                start_t = time.time()
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                step_t = time.time() - start_t
                total_time += step_t

                loss_val = loss.item()
                curve.append((step, loss_val))
                logging.info(f"[{scenario_name} seed {seed_idx}] step {step+1}/{total_steps} | loss {loss_val:.4f} | time {step_t:.3f}s")
                step += 1

                # Batch size is 1, so samples consumed == step.
                if eval_batches and eval_every > 0 and step % eval_every == 0:
                    ev = _eval_loss(model, eval_batches, accelerator.device)
                    if ev is not None:
                        eval_curve.append((step, ev))
                        logging.info(f"[{scenario_name} seed {seed_idx}] eval@{step} "
                                     f"| eval_loss {ev:.4f}")

        # Always finish on an eval point, so final-loss comparisons are made at
        # the same amount of training as every other arm.
        if eval_batches and eval_every > 0 and (not eval_curve or eval_curve[-1][0] != step):
            ev = _eval_loss(model, eval_batches, accelerator.device)
            if ev is not None:
                eval_curve.append((step, ev))

        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        avg_time_ms = (total_time / total_steps) * 1000 if total_steps > 0 else 0

        # Second, independent offload check. The structural assert above proves the
        # partitions live on CPU; this proves the whole model was never resident on
        # GPU at once. If peak ever approaches the full bf16 footprint, ZeRO-3 was
        # gathering far more than it should and the throughput numbers are not
        # measuring an offload workload.
        logging.info(
            f"Peak GPU: {peak_mem_mb:.1f} MiB vs {model_bf16_mib:.1f} MiB full model "
            f"({100*peak_mem_mb/model_bf16_mib:.1f}% resident-equivalent)"
        )
        assert peak_mem_mb < model_bf16_mib, (
            f"Peak GPU memory ({peak_mem_mb:.1f} MiB) reached the full bf16 model size "
            f"({model_bf16_mib:.1f} MiB) — nothing was meaningfully offloaded."
        )

        res = {
            "curve": curve,
            "eval_curve": eval_curve,
            "metrics": {
                "peak_gpu_mb": float(peak_mem_mb),
                "step_time_ms": float(avg_time_ms),
                # Evidence that this really was an offload run, kept alongside the
                # numbers it qualifies.
                "gpu_budget_mib": int(budget_mib),
                "model_bf16_mib": float(model_bf16_mib),
                "offloaded_to_cpu_mib": float(offloaded_mib),
                # Load-phase transient, higher than peak_gpu_mb. This is the number
                # that constrains how low gpu_budget_mib can go.
                "peak_load_mb": float(peak_load_mb),
                "trainable_params": int(n_trainable),
                "total_params": int(n_total),
            }
        }

        with open(out_file, "w") as f:
            json.dump(res, f, indent=2)
            
        logging.info("Training complete, results saved.")
        
    except Exception as e:
        logging.error(f"DeepSpeed initialization or training failed: {e}", exc_info=True)
        res = {"error": str(e)}
        with open(out_file, "w") as f:
            json.dump(res, f, indent=2)
        # Exit non-zero so the orchestrator's returncode check actually fires.
        sys.exit(1)

if __name__ == "__main__":
    main()
