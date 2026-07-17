import json
from typing import Dict, Any
from .runners import BaseRunner, OpenAIRunner

class Evaluator:
    def evaluate(self, prompt: str, baseline_response: str, test_response: str) -> Dict[str, Any]:
        raise NotImplementedError

class ExactMatchEvaluator(Evaluator):
    """Compares the exact string or token output. Useful for checking if quantization changes behavior."""
    def evaluate(self, prompt: str, baseline_response: str, test_response: str) -> Dict[str, Any]:
        is_match = (baseline_response.strip() == test_response.strip())
        return {
            "exact_match": is_match,
            "baseline_len": len(baseline_response),
            "test_len": len(test_response)
        }

class PerformanceEvaluator(Evaluator):
    """Compares latency and tokens/sec between baseline and test to show speedup and implied bandwidth saving."""
    def evaluate(self, prompt: str, baseline_result: Dict[str, Any], test_result: Dict[str, Any]) -> Dict[str, Any]:
        b_tps = baseline_result.get("tokens_per_sec", 0)
        t_tps = test_result.get("tokens_per_sec", 0)
        
        speedup = (t_tps / b_tps) if b_tps > 0 else 0
        
        return {
            "baseline_latency_sec": baseline_result.get("latency_sec", 0),
            "test_latency_sec": test_result.get("latency_sec", 0),
            "baseline_ttft_sec": baseline_result.get("ttft_sec", 0),
            "test_ttft_sec": test_result.get("ttft_sec", 0),
            "baseline_tokens_per_sec": b_tps,
            "test_tokens_per_sec": t_tps,
            "speedup_x": speedup
        }

class SemanticSimilarityEvaluator(Evaluator):
    """Uses a sentence transformer model to compute the cosine similarity between the two responses."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def evaluate(self, prompt: str, baseline_response: str, test_response: str) -> Dict[str, Any]:
        from sentence_transformers import util
        embeddings = self.model.encode([baseline_response, test_response])
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return {
            "semantic_similarity": similarity
        }

class LLMJudgeEvaluator(Evaluator):
    """Uses a strong LLM (e.g. GPT-4) to judge the quality of the test response compared to the baseline."""
    def __init__(self, judge_model_id: str = "gpt-4", api_key: str = "", base_url: str = "https://api.openai.com"):
        self.judge_model_id = judge_model_id
        self.runner = OpenAIRunner(base_url=base_url, api_key=api_key)
        self.system_prompt = (
            "You are an impartial judge evaluating the quality of two AI assistant responses to a user prompt. "
            "You will be given the User Prompt, a Baseline Response, and a Test Response. "
            "Evaluate the Test Response based on helpfulness, relevance, accuracy, and detail compared to the Baseline Response. "
            "Output your evaluation strictly in JSON format with two keys: "
            "'score': an integer from 1 to 10 for the test response quality (where 10 is much better than baseline, 5 is equal, 1 is much worse). "
            "'reasoning': a short explanation of your score."
        )

    def evaluate(self, prompt: str, baseline_response: str, test_response: str) -> Dict[str, Any]:
        user_message = (
            f"User Prompt:\n{prompt}\n\n"
            f"Baseline Response:\n{baseline_response}\n\n"
            f"Test Response:\n{test_response}\n\n"
            "Evaluate the Test Response. Provide JSON."
        )
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        try:
            judge_output_dict = self.runner.generate(self.judge_model_id, messages, max_tokens=256, temperature=0.0)
            judge_output = judge_output_dict["text"]
            
            # Clean up potential markdown formatting in output
            if judge_output.startswith("```json"):
                judge_output = judge_output.replace("```json", "", 1)
            if judge_output.endswith("```"):
                judge_output = judge_output.rsplit("```", 1)[0]
                
            result = json.loads(judge_output.strip())
            return {
                "judge_score": result.get("score", 0),
                "judge_reasoning": result.get("reasoning", "")
            }
        except Exception as e:
            return {
                "judge_score": 0,
                "judge_reasoning": f"Failed to evaluate: {str(e)}"
            }
