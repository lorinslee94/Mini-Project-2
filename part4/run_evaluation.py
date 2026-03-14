# part4/run_evaluation.py

import os
import json

from openai import OpenAI

from part3.head_agent import Head_Agent
from part4.dataset_generator import TestDatasetGenerator
from part4.judge_and_eval import LLM_Judge, EvaluationPipeline

# -----------------------------
# 1. Check environment variables
# -----------------------------
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

PINECONDE_INDEX = "ml-textbook-rag-1536"

if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
if not pinecone_api_key:
    raise RuntimeError("PINECONE_API_KEY environment variable is not set.")

# -----------------------------
# 2. Initialize OpenAI client
# -----------------------------
client = OpenAI(api_key=openai_api_key)

# -----------------------------
# 3. Load or generate test_set.json
# -----------------------------
current_dir = os.path.dirname(__file__)
test_set_path = os.path.join(current_dir, "test_set.json")

generator = TestDatasetGenerator(client)

if not os.path.exists(test_set_path):
    print("test_set.json not found. Generating a new dataset...")
    generator.build_full_dataset()
    generator.save_dataset(test_set_path)
    print(f"Saved test_set.json to {test_set_path}")

data = generator.load_dataset(test_set_path)

# -----------------------------
# 4. Initialize Head_Agent (from Part 3)
# -----------------------------
head_agent = Head_Agent(
    openai_key=openai_api_key,
    pinecone_key=pinecone_api_key,
    pinecone_index_name=PINECONDE_INDEX
)

# -----------------------------
# 5. Initialize Judge + EvaluationPipeline
# -----------------------------
judge = LLM_Judge(client)
pipeline = EvaluationPipeline(head_agent, judge)

# -----------------------------
# 6. Run single-turn categories
# -----------------------------
print("Running single-turn tests...")

pipeline.run_single_turn_test("obnoxious", data["obnoxious"])
pipeline.run_single_turn_test("irrelevant", data["irrelevant"])
pipeline.run_single_turn_test("relevant", data["relevant"])
pipeline.run_single_turn_test("small_talk", data["small_talk"])
pipeline.run_single_turn_test("hybrid", data["hybrid"])

# -----------------------------
# 7. Run multi-turn category
# -----------------------------
print("Running multi-turn tests...")
pipeline.run_multi_turn_test(data["multi_turn"])

# -----------------------------
# 8. Compute and print metrics
# -----------------------------
summary, overall_acc = pipeline.calculate_metrics()

# -----------------------------
# 9. Save evaluation results for analysis/PDF
# -----------------------------
results_path = os.path.join(current_dir, "eval_results.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "summary": summary,
            "overall_accuracy": overall_acc,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )

print(f"Saved eval_results.json to {results_path}")

