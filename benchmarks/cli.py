import argparse
import json
import sys
import os
from .runners import OpenAIRunner, TransformersRunner
from .datasets import get_dataset
from .evaluators import ExactMatchEvaluator, SemanticSimilarityEvaluator, LLMJudgeEvaluator

def main():
    parser = argparse.ArgumentParser(description="Kiwi LLM Quality Benchmarking Tool")
    parser.add_argument("--model", type=str, required=True, help="Model ID to evaluate")
    parser.add_argument("--baseline-model", type=str, default=None, help="Model ID for the baseline (defaults to --model)")
    parser.add_argument("--dataset", type=str, default="dummy", help="Dataset name to use (dummy or huggingface repo)")
    parser.add_argument("--limit", type=int, default=5, help="Number of prompts to evaluate")
    parser.add_argument("--test-url", type=str, default="http://localhost:8080", help="URL for the Kiwi gateway or test endpoint")
    parser.add_argument("--baseline-url", type=str, default=None, help="URL for the baseline endpoint (if not using local Transformers)")
    parser.add_argument("--use-transformers-baseline", action="store_true", help="Use local Transformers model as baseline")
    parser.add_argument("--evaluators", type=str, default="exact_match,performance", help="Comma-separated list of evaluators: exact_match,semantic,llm_judge,performance")
    parser.add_argument("--judge-api-key", type=str, default=os.getenv("OPENAI_API_KEY", ""), help="API Key for LLM Judge")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Output JSON file for results")
    
    args = parser.parse_args()

    # Initialize Runners
    print(f"Initializing test runner at {args.test_url}...")
    test_runner = OpenAIRunner(base_url=args.test_url)
    
    if args.use_transformers_baseline:
        print("Initializing local Transformers baseline (this may take a while to load model)...")
        baseline_runner = TransformersRunner()
    elif args.baseline_url:
        print(f"Initializing baseline runner at {args.baseline_url}...")
        baseline_runner = OpenAIRunner(base_url=args.baseline_url)
    else:
        print("Error: Must specify either --baseline-url or --use-transformers-baseline")
        sys.exit(1)

    # Initialize Evaluators
    eval_names = args.evaluators.split(",")
    evaluators = []
    if "exact_match" in eval_names:
        evaluators.append(("Exact Match", ExactMatchEvaluator()))
    if "semantic" in eval_names:
        print("Loading Semantic Similarity Evaluator (will download sentence-transformer model if first time)...")
        evaluators.append(("Semantic", SemanticSimilarityEvaluator()))
    if "llm_judge" in eval_names:
        if not args.judge_api_key:
            print("Warning: llm_judge requested but no --judge-api-key or OPENAI_API_KEY provided.")
        evaluators.append(("LLM Judge", LLMJudgeEvaluator(api_key=args.judge_api_key)))
    if "performance" in eval_names:
        from .evaluators import PerformanceEvaluator
        evaluators.append(("Performance", PerformanceEvaluator()))

    # Load Dataset
    print(f"Loading dataset '{args.dataset}' (limit: {args.limit})...")
    prompts = get_dataset(args.dataset, limit=args.limit)

    results = []
    
    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        print(f"\n[{i+1}/{len(prompts)}] Evaluating prompt: {prompt[:50]}...")
        
        messages = [{"role": "user", "content": prompt}]
        
        # Generate baseline
        print("  Generating baseline response...")
        try:
            b_model = args.baseline_model if args.baseline_model else args.model
            baseline_result = baseline_runner.generate(b_model, messages)
            baseline_response = baseline_result["text"]
        except Exception as e:
            print(f"  Failed baseline generation: {e}")
            baseline_result = {}
            baseline_response = ""

        # Generate test
        print("  Generating test response...")
        try:
            test_result = test_runner.generate(args.model, messages)
            test_response = test_result["text"]
        except Exception as e:
            print(f"  Failed test generation: {e}")
            test_result = {}
            test_response = ""

        # Evaluate
        eval_metrics = {}
        for name, evaluator in evaluators:
            print(f"  Running evaluator: {name}...")
            if name == "Performance":
                metrics = evaluator.evaluate(prompt, baseline_result, test_result)
            else:
                metrics = evaluator.evaluate(prompt, baseline_response, test_response)
            eval_metrics.update(metrics)

        result_item = {
            "prompt": prompt,
            "baseline_response": baseline_response,
            "test_response": test_response,
            "metrics": eval_metrics
        }
        results.append(result_item)
        print(f"  Metrics: {eval_metrics}")

    # Summarize
    print(f"\nSaving results to {args.output}")
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
        
    print("Benchmark complete!")

if __name__ == "__main__":
    main()
