import httpx
from typing import List, Dict, Optional, Any
import time

class BaseRunner:
    """Base class for executing inference requests."""
    def generate(self, model_id: str, messages: List[Dict[str, str]], max_tokens: int = 128, temperature: float = 0.0) -> Dict[str, Any]:
        raise NotImplementedError

class OpenAIRunner(BaseRunner):
    """
    Runner for any OpenAI-compatible endpoint. 
    This can be used for both the Kiwi Gateway and a vLLM/Ollama baseline.
    """
    def __init__(self, base_url: str, api_key: str = "dummy"):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        
    def generate(self, model_id: str, messages: List[Dict[str, str]], max_tokens: int = 128, temperature: float = 0.0) -> Dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        start_time = time.time()
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            
            end_time = time.time()
            latency = end_time - start_time
            
            text = ""
            if "choices" in data and len(data["choices"]) > 0:
                if "message" in data["choices"][0]:
                    text = data["choices"][0]["message"].get("content", "")
                elif "delta" in data["choices"][0]:
                    text = data["choices"][0]["delta"].get("content", "")
            
            # Simple heuristic: ~4 characters per token
            tokens = len(text) / 4.0
            
            return {
                "text": text,
                "latency_sec": latency,
                "tokens_generated": tokens,
                "tokens_per_sec": tokens / latency if latency > 0 else 0
            }

class TransformersRunner(BaseRunner):
    """
    Runner for local HuggingFace transformers baseline (runs locally on single GPU).
    Lazy-loads the model so it doesn't consume memory until needed.
    """
    def __init__(self, dtype="bfloat16"):
        self.dtype = dtype
        self._models = {}
        self._tokenizers = {}
        
    def _load(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if model_id not in self._models:
            dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
            self._tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)
            self._models[model_id] = AutoModelForCausalLM.from_pretrained(
                model_id, 
                torch_dtype=dtype_map.get(self.dtype, torch.bfloat16), 
                device_map="auto"
            )
            self._models[model_id].eval()
            
    def generate(self, model_id: str, messages: List[Dict[str, str]], max_tokens: int = 128, temperature: float = 0.0) -> Dict[str, Any]:
        self._load(model_id)
        import torch
        import time
        tokenizer = self._tokenizers[model_id]
        model = self._models[model_id]
        
        text = ""
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            # Fallback for models like gpt2 that don't have a chat template
            for msg in messages:
                text += f"{msg['role']}: {msg['content']}\n"
            text += "assistant: "
            
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        start_time = time.time()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_tokens, 
                temperature=temperature,
                do_sample=(temperature > 0.0),
                pad_token_id=tokenizer.eos_token_id
            )
        end_time = time.time()
        latency = end_time - start_time
            
        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        tokens_generated = len(generated_ids)
        
        return {
            "text": generated_text,
            "latency_sec": latency,
            "tokens_generated": tokens_generated,
            "tokens_per_sec": tokens_generated / latency if latency > 0 else 0
        }
