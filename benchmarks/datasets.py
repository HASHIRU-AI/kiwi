import json
from typing import List, Dict

class DatasetLoader:
    def load(self, split: str = "test", limit: int = None) -> List[Dict[str, str]]:
        raise NotImplementedError

class DummyDatasetLoader(DatasetLoader):
    """A simple dataset loader with a few static prompts for quick testing."""
    def load(self, split: str = "test", limit: int = None) -> List[Dict[str, str]]:
        data = [
            {"prompt": "Write a Python script to sort a list of integers."},
            {"prompt": "Explain the difference between deep learning and machine learning."},
            {"prompt": "What are the key benefits of distributed inference for LLMs?"},
        ]
        return data[:limit] if limit else data

class HuggingFaceDatasetLoader(DatasetLoader):
    """Loads a dataset from the HuggingFace Hub. e.g. 'tatsu-lab/alpaca' or 'lmsys/mt_bench_human_judgments'"""
    def __init__(self, dataset_name: str, text_column: str = "instruction"):
        self.dataset_name = dataset_name
        self.text_column = text_column

    def load(self, split: str = "test", limit: int = None) -> List[Dict[str, str]]:
        from datasets import load_dataset
        # Some datasets don't have a test split, we try falling back to train
        try:
            ds = load_dataset(self.dataset_name, split=split)
        except ValueError:
            ds = load_dataset(self.dataset_name, split="train")
            
        if limit:
            ds = ds.select(range(min(limit, len(ds))))
            
        return [{"prompt": item[self.text_column]} for item in ds]

def get_dataset(name: str, limit: int = None) -> List[Dict[str, str]]:
    if name == "dummy":
        return DummyDatasetLoader().load(limit=limit)
    else:
        # Default fallback to assuming it's a HuggingFace dataset
        return HuggingFaceDatasetLoader(name).load(limit=limit)
