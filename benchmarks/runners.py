import httpx
from typing import List, Dict, Optional
import time

class BaseRunner:
    """Base class for executing inference requests."""
    def generate(self, model_id: str, messages: List[Dict[str, str]], max_tokens: int = 128, temperature: float = 0.0) -> str:
        raise NotImplementedError

class OpenAIRunner(BaseRunner):
    """
    Runner for any OpenAI-compatible endpoint. 
    This can be used for both the Kiwi Gateway and a vLLM/Ollama baseline.
    """
    def __init__(self, base_url: str, api_key: str = "dummy"):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        
    def generate(self, model_id: str, messages: List[Dict[str, str]], max_tokens: int = 128, temperature: float = 0.0) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                if "message" in data["choices"][0]:
                    return data["choices"][0]["message"].get("content", "")
                elif "delta" in data["choices"][0]:
                    return data["choices"][0]["delta"].get("content", "")
            return ""

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
            
    def generate(self, model_id: str, messages: List[Dict[str, str]], max_tokens: int = 128, temperature: float = 0.0) -> str:
        self._load(model_id)
        import torch
        tokenizer = self._tokenizers[model_id]
        model = self._models[model_id]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_tokens, 
                temperature=temperature,
                do_sample=(temperature > 0.0),
                pad_token_id=tokenizer.eos_token_id
            )
            
        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True)
