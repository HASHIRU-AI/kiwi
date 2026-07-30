import time
import httpx
import torch
import os
from typing import List, Tuple, Dict, Any
from safetensors.torch import save as safetensors_save, load as safetensors_load
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

class BaseRunner:
    def run_training(self, model_id: str, dataset: List[Dict[str, Any]], epochs: int, scenario_name: str = "unknown", seed_idx: int = 0) -> List[Tuple[int, float]]:
        """Runs training and returns a list of (step, loss) tuples."""
        raise NotImplementedError

class SingleMachineRunner(BaseRunner):
    def run_training(self, model_id: str, dataset: List[Dict[str, Any]], epochs: int, scenario_name: str = "unknown", seed_idx: int = 0) -> List[Tuple[int, float]]:
        from peft import LoraConfig, get_peft_model
        
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
        
        # Apply PEFT LoRA with the exact config used in Kiwi workers
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        model.train()
        
        kiwi_lr = float(os.getenv("KIWI_LR", "1e-5"))
        logging.info(f"Using learning rate: {kiwi_lr}")
        optimizer = torch.optim.Adam(model.parameters(), lr=kiwi_lr)
        
        total_steps = epochs * len(dataset)
        curve = []
        step = 0
        for epoch in range(epochs):
            for sample in dataset:
                input_ids = sample["input_ids"].to(model.device)
                labels = sample["labels"].to(model.device)
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                loss_val = loss.item()
                curve.append((step, loss_val))
                logging.info(f"[{scenario_name} seed {seed_idx}] step {step+1}/{total_steps} | loss {loss_val:.4f}")
                step += 1
                
        # cleanup
        del model
        del optimizer
        torch.cuda.empty_cache()
        
        return curve

class DistributedRunner(BaseRunner):
    def __init__(self, gateway_url: str = "http://localhost:8080"):
        self.gateway_url = gateway_url

    def run_training(self, model_id: str, dataset: List[Dict[str, Any]], epochs: int, scenario_name: str = "unknown", seed_idx: int = 0) -> List[Tuple[int, float]]:
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
        for epoch in range(epochs):
            for sample in dataset:
                out = post(sample["payload"], f"step-{step}")
                loss_val = out["loss"].item()
                curve.append((step, loss_val))
                logging.info(f"[{scenario_name} seed {seed_idx}] step {step+1}/{total_steps} | loss {loss_val:.4f}")
                step += 1
                
        return curve
