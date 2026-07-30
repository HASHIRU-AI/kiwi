import gc
import time
import httpx
import torch
import os
from typing import List, Tuple, Dict, Any
from safetensors.torch import save as safetensors_save, load as safetensors_load
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

SHARED_TRAIN_CONFIG = {
    "lora_config": LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    ),
    "lr": float(os.getenv("KIWI_LR", "1e-5"))
}

class BaseRunner:
    def run_training(self, model_id: str, dataset: List[Dict[str, Any]], epochs: int, scenario_name: str = "unknown", seed_idx: int = 0) -> Dict[str, Any]:
        """Runs training and returns a dict with 'curve' and 'metrics'."""
        raise NotImplementedError

class SingleMachineRunner(BaseRunner):
    def run_training(self, model_id: str, dataset: List[Dict[str, Any]], epochs: int, scenario_name: str = "unknown", seed_idx: int = 0) -> Dict[str, Any]:
        device = "cuda"
        dtype = torch.bfloat16
        logging.info(f"Using device: {device}")
        logging.info(f"Using dtype: {dtype}")
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=dtype, 
            device_map=device
        )
        
        model = get_peft_model(model, SHARED_TRAIN_CONFIG["lora_config"])
        model.train()
        
        logging.info(f"Using learning rate: {SHARED_TRAIN_CONFIG['lr']}")
        optimizer = torch.optim.Adam(model.parameters(), lr=SHARED_TRAIN_CONFIG["lr"])
        
        torch.cuda.reset_peak_memory_stats()
        
        total_steps = epochs * len(dataset)
        curve = []
        step = 0
        total_time = 0.0
        
        for epoch in range(epochs):
            for sample in dataset:
                input_ids = sample["input_ids"].to(model.device)
                labels = sample["labels"].to(model.device)
                
                start_t = time.time()
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                step_t = time.time() - start_t
                total_time += step_t
                
                loss_val = loss.item()
                curve.append((step, loss_val))
                logging.info(f"[{scenario_name} seed {seed_idx}] step {step+1}/{total_steps} | loss {loss_val:.4f} | time {step_t:.3f}s")
                step += 1
                
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        avg_time_ms = (total_time / total_steps) * 1000 if total_steps > 0 else 0
        
        # cleanup
        del model
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            "curve": curve,
            "metrics": {
                "peak_gpu_mb": float(peak_mem_mb),
                "step_time_ms": float(avg_time_ms)
            }
        }

class DistributedRunner(BaseRunner):
    def __init__(self, gateway_url: str = "http://localhost:8080"):
        self.gateway_url = gateway_url

    def run_training(self, model_id: str, dataset: List[Dict[str, Any]], epochs: int, scenario_name: str = "unknown", seed_idx: int = 0) -> Dict[str, Any]:
        def post(payload, rid):
            r = httpx.post(f"{self.gateway_url}/train_step", content=payload,
                           headers={"Content-Type": "application/octet-stream",
                                    "X-Model-ID": model_id,
                                    "X-Request-ID": rid},
                           timeout=120.0)
            r.raise_for_status()
            return safetensors_load(r.content)

        total_steps = epochs * len(dataset)
        curve = []
        step = 0
        total_time = 0.0
        
        for epoch in range(epochs):
            for sample in dataset:
                start_t = time.time()
                out = post(sample["payload"], f"step-{step}")
                step_t = time.time() - start_t
                total_time += step_t
                
                loss_val = out["loss"].item()
                curve.append((step, loss_val))
                logging.info(f"[{scenario_name} seed {seed_idx}] step {step+1}/{total_steps} | loss {loss_val:.4f} | time {step_t:.3f}s")
                step += 1
                
        avg_time_ms = (total_time / total_steps) * 1000 if total_steps > 0 else 0
        
        return {
            "curve": curve,
            "metrics": {
                "peak_gpu_mb": 0.0,
                "step_time_ms": float(avg_time_ms)
            }
        }

# class OffloadRunner(BaseRunner):
#     def run_training(self, model_id: str, dataset: List[Dict[str, Any]], epochs: int, scenario_name: str = "unknown", seed_idx: int = 0) -> Dict[str, Any]:
#         dtype = torch.bfloat16
#         logging.info(f"Using dtype: {dtype} with Offload (512MiB GPU limit, remainder to CPU)")
#         
#         model = AutoModelForCausalLM.from_pretrained(
#             model_id, 
#             torch_dtype=dtype, 
#             device_map="auto",
#             max_memory={0: "512MiB", "cpu": "16GiB"}
#         )
#         
#         logging.info(f"Model hf_device_map: {model.hf_device_map}")
#         gpu_layers = [k for k, v in model.hf_device_map.items() if v == 0]
#         cpu_layers = [k for k, v in model.hf_device_map.items() if v == "cpu"]
#         logging.info(f"  GPU layers ({len(gpu_layers)}): {gpu_layers[:5]}{'...' if len(gpu_layers) > 5 else ''}")
#         logging.info(f"  CPU layers ({len(cpu_layers)}): {cpu_layers[:5]}{'...' if len(cpu_layers) > 5 else ''}")
#         assert len(cpu_layers) > 0, (
#             f"OffloadRunner: no layers were offloaded to CPU — "
#             f"the model fits entirely in {512}MiB. Tighten max_memory further."
#         )
#         
#         # Required when using device_map="auto" with PEFT — without this,
#         # accelerate's meta-device tensors for CPU-offloaded layers cause a
#         # device mismatch error ("expected device meta but got cuda:0") during
#         # the backward pass through LoRA adapters.
#         model.enable_input_require_grads()
#         model = get_peft_model(model, SHARED_TRAIN_CONFIG["lora_config"])
#         model.train()
#         
#         logging.info(f"Using learning rate: {SHARED_TRAIN_CONFIG['lr']}")
#         optimizer = torch.optim.Adam(model.parameters(), lr=SHARED_TRAIN_CONFIG["lr"])
#         
#         if len(dataset) > 0:
#             sample = dataset[0]
#             first_device = next(model.parameters()).device
#             input_ids = sample["input_ids"].to(first_device)
#             labels = sample["labels"].to(first_device)
#             
#             optimizer.zero_grad()
#             outputs = model(input_ids=input_ids, labels=labels)
#             loss = outputs.loss
#             loss.backward()
#             
#             has_grads = any(p.grad is not None for p in model.parameters())
#             assert has_grads, "Dummy step failed: all gradients are None (possible device_map bug)."
#             optimizer.zero_grad()
#             logging.info("Dummy step passed, gradients are flowing.")
#             
#         torch.cuda.reset_peak_memory_stats()
#         
#         total_steps = epochs * len(dataset)
#         curve = []
#         step = 0
#         total_time = 0.0
#         
#         for epoch in range(epochs):
#             for sample in dataset:
#                 first_device = next(model.parameters()).device
#                 input_ids = sample["input_ids"].to(first_device)
#                 labels = sample["labels"].to(first_device)
#                 
#                 start_t = time.time()
#                 optimizer.zero_grad()
#                 outputs = model(input_ids=input_ids, labels=labels)
#                 loss = outputs.loss
#                 loss.backward()
#                 optimizer.step()
#                 step_t = time.time() - start_t
#                 total_time += step_t
#                 
#                 loss_val = loss.item()
#                 curve.append((step, loss_val))
#                 logging.info(f"[{scenario_name} seed {seed_idx}] step {step+1}/{total_steps} | loss {loss_val:.4f} | time {step_t:.3f}s")
#                 step += 1
#                 
#         peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
#         avg_time_ms = (total_time / total_steps) * 1000 if total_steps > 0 else 0
#         
#         # cleanup
#         del model
#         del optimizer
#         gc.collect()
#         torch.cuda.empty_cache()
#         
#         return {
#             "curve": curve,
#             "metrics": {
#                 "peak_gpu_mb": float(peak_mem_mb),
#                 "step_time_ms": float(avg_time_ms)
#             }
#         }

_qlora_fraction_set = False

class QLoRARunner(BaseRunner):
    def run_training(self, model_id: str, dataset: List[Dict[str, Any]], epochs: int, scenario_name: str = "unknown", seed_idx: int = 0) -> Dict[str, Any]:
        global _qlora_fraction_set
        
        try:
            import bitsandbytes as bnb
            from transformers import BitsAndBytesConfig
            from peft import prepare_model_for_kbit_training
        except ImportError:
            logging.error("bitsandbytes not installed or not working. Please install bitsandbytes to run QLoRARunner.")
            raise
        
        # Ensure previous run's memory is fully released before loading
        gc.collect()
        torch.cuda.empty_cache()
            
        total_vram = torch.cuda.get_device_properties(0).total_memory
        target_bytes = 1024 * 1024 * 1024 # 1GB
        fraction = min(1.0, target_bytes / total_vram)
        
        # set_per_process_memory_fraction is process-global and sticky;
        # only call it once to avoid errors on subsequent seeds.
        if not _qlora_fraction_set:
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
            _qlora_fraction_set = True
        logging.info(f"QLoRA memory fraction {fraction:.4f} (~1GB of {total_vram/1024/1024/1024:.1f}GB total VRAM)")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="cuda"
        )
        
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model = get_peft_model(model, SHARED_TRAIN_CONFIG["lora_config"])
        model.train()
        
        logging.info(f"Using learning rate: {SHARED_TRAIN_CONFIG['lr']}")
        optimizer = torch.optim.Adam(model.parameters(), lr=SHARED_TRAIN_CONFIG["lr"])
        
        if len(dataset) > 0:
            sample = dataset[0]
            input_ids = sample["input_ids"].to(model.device)
            labels = sample["labels"].to(model.device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            assert not torch.isnan(loss), "Dummy step failed: loss is NaN."
            optimizer.zero_grad()
            logging.info("Dummy step passed, loss is valid.")
            
        torch.cuda.reset_peak_memory_stats()
        
        total_steps = epochs * len(dataset)
        curve = []
        step = 0
        total_time = 0.0
        
        for epoch in range(epochs):
            for sample in dataset:
                input_ids = sample["input_ids"].to(model.device)
                labels = sample["labels"].to(model.device)
                
                start_t = time.time()
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                step_t = time.time() - start_t
                total_time += step_t
                
                loss_val = loss.item()
                curve.append((step, loss_val))
                logging.info(f"[{scenario_name} seed {seed_idx}] step {step+1}/{total_steps} | loss {loss_val:.4f} | time {step_t:.3f}s")
                step += 1
                
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        avg_time_ms = (total_time / total_steps) * 1000 if total_steps > 0 else 0
        
        # cleanup
        del model
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
        
        return {
            "curve": curve,
            "metrics": {
                "peak_gpu_mb": float(peak_mem_mb),
                "step_time_ms": float(avg_time_ms)
            }
        }
