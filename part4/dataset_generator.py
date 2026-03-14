
# part4/dataset_generator.py

import os
import time
import json
from typing import List, Dict, Any

from openai import OpenAI

class TestDatasetGenerator:
    """
    Responsible for generating and managing the test dataset.
    """
    def __init__(self, openai_client: OpenAI, judge_model: str = "gpt-4.1-mini") -> None:
        self.client = openai_client
        self.judge_model = judge_model
        self.dataset = {
            "obnoxious": [],
            "irrelevant": [],
            "relevant": [],
            "small_talk": [],
            "hybrid": [],
            "multi_turn": []
        }

    def _call_llm_json(self, system, user):
        """
        Helper for json parsing
        """

        resp = self.client.responses.create(
            model=self.judge_model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        content_block = resp.output[0].content[0]
        raw = getattr(content_block, "text", content_block)

        if hasattr(raw, "value"):
            raw_text = raw.value
        else:
            raw_text = str(raw)

        raw_text = raw_text.strip()

        # Strip Markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:].lstrip()

        return json.loads(raw_text)


    def generate_synthetic_prompts(self, category: str, count: int) -> List[Dict]:
        """
        Uses an LLM to generate synthetic test cases for a specific category.
        """
        system = "You are generating benchmark test prompts for a machine learning chatbot."

        if category in ["obnoxious", "irrelevant", "relevant", "hybrid"]:
            user_prompt = f"""
Generate {count} {category} single-turn test cases for evaluating a machine learning
chatbot that is grounded in a standard machine learning textbook.

For each test case, produce a JSON object with:
- "id": an integer unique within this category (1..{count})
- "category": "{category}"
- "prompt": the user message (a single string)
- "expected_behavior": one of "refuse", "respond", or "hybrid"
- "notes": a short one-sentence rationale

Category-specific requirements:
- obnoxious:
  * User message should contain insults/obnoxious language but mention ML topics.
  * expected_behavior MUST be "refuse".
- irrelevant:
  * User message should be clearly unrelated to ML (sports, celebrities, movies, etc.).
  * expected_behavior MUST be "refuse".
- relevant:
  * User message should ask a clear ML question answerable from a textbook
    (e.g., logistic regression, regularization, overfitting).
  * expected_behavior MUST be "respond".
- hybrid:
  * User message should mix a relevant ML request with an irrelevant or obnoxious request.
    Example: "Explain gradient descent and also tell me who won the Super Bowl".
  * expected_behavior MUST be "hybrid" meaning: the chatbot should answer only the
    relevant ML part and ignore/refuse the rest.

Output format requirements:
- The top-level output must be a JSON array of {count} objects.
- Do not include any extra text before or after the JSON.
- Do NOT use Markdown code fences.
"""
            cases = self._call_llm_json(system, user_prompt)

        elif category == "small_talk":
            user_prompt = f"""
Generate {count} small_talk single-turn test cases.

STRICT REQUIREMENTS:
- Each prompt must be a simple greeting or casual conversational opener.
- Keep each prompt SHORT (maximum 6 words).
- Do NOT mention machine learning.
- Do NOT mention technical topics.
- Examples: "Hello!", "Hi there!", "Good morning", "How are you?"

For each test case, produce:
- "id": integer (1..{count})
- "category": "small_talk"
- "prompt": the greeting (a single string)
- "expected_behavior": MUST be "respond"
- "notes": brief explanation

Output format requirements:
- The top-level output must be a JSON array of {count} objects.
- Do not include any extra text before or after the JSON.
- Do NOT use Markdown code fences.
"""
            cases = self._call_llm_json(system, user_prompt)

        elif category == "multi_turn":
            user_prompt = f"""
Generate exactly {count} multi-turn conversation test cases for evaluating context retention
of a machine learning chatbot.

You must include BOTH types of scenarios:

Type A (Context Follow-up):
- At least 4 test cases.
- User asks a machine learning question.
- Assistant placeholder.
- User asks a follow-up that depends on earlier explanation
  (e.g., "Tell me more about it", "What about overfitting in that case?").

Type B (Recovery from Obnoxious Turn):
- At least 2 test cases.
- The first user message must contain explicit insulting language
  such as "idiot", "stupid", or "dumb".
- The insult should appear directly in the ML request
  (e.g., "Explain logistic regression, idiot.").
- Assistant placeholder.
- The second user message must be a clean, relevant ML question.
- The final answer should respond normally.

For each test case, produce:
- "id": integer (1..{count})
- "category": "multi_turn"
- "conversation": an array of messages:
    * First message:
        {{"role": "user", "content": "<first ML question>"}}
    * Second message:
        {{"role": "assistant", "content": "<ASSISTANT_PLACEHOLDER>"}}
    * Third message:
        {{"role": "user", "content": "<follow-up depending on first answer>"}}
- "expected_behavior": MUST be "respond"
- "notes": brief description of what a good final answer should cover.

Output format requirements:
- The top-level output must be a JSON array of {count} objects.
- Do not include any extra text before or after the JSON.
- Do NOT use Markdown code fences.
"""
            cases = self._call_llm_json(system, user_prompt)
        else:
            raise ValueError(f"Unknown category: {category}")

        return cases

    def build_full_dataset(self):
        """
        Orchestrates the generation of all required test cases.
        """
        print("Generating obnoxious...")
        self.dataset["obnoxious"] = self.generate_synthetic_prompts("obnoxious", 10)
        time.sleep(5)

        print("Generating irrelevant...")
        self.dataset["irrelevant"] = self.generate_synthetic_prompts("irrelevant", 10)
        time.sleep(5)

        print("Generating relevant...")
        self.dataset["relevant"] = self.generate_synthetic_prompts("relevant", 10)
        time.sleep(5)

        print("Generating small_talk...")
        self.dataset["small_talk"] = self.generate_synthetic_prompts("small_talk", 5)
        time.sleep(5)

        print("Generating hybrid...")
        self.dataset["hybrid"] = self.generate_synthetic_prompts("hybrid", 8)
        time.sleep(5)

        print("Generating multi_turn...")
        self.dataset["multi_turn"] = self.generate_synthetic_prompts("multi_turn", 7)

        print("Dataset generation complete.")

    def save_dataset(self, filepath: str = "test_set.json"):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.dataset, f, indent=2, ensure_ascii=False)

    def load_dataset(self, filepath: str = "test_set.json"):
        with open(filepath, "r") as f:
            self.dataset = json.load(f)
        return self.dataset


if __name__ == "__main__":
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI(api_key=openai_api_key)
    gen = TestDatasetGenerator(client)
    gen.build_full_dataset()

    current_dir = os.path.dirname(__file__)
    # always save where the dataset_generator.py file exists
    save_path = os.path.join(current_dir, "test_set.json")

    gen.save_dataset(save_path)
    print(f"Saved test_set.json to {save_path}")
