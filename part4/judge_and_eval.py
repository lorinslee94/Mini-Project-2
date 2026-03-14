
# part4/judge_and_eval.py

from typing import List, Dict, Any, Tuple

from openai import OpenAI
from part3.head_agent import Head_Agent


class LLM_Judge:
    """
    The 'LLM-as-a-Judge' that evaluates the chatbot's performance.
    """

    def __init__(self, openai_client) -> None:
        self.client = openai_client

    def construct_judge_prompt(self, user_input, bot_response, category):
        """
        Constructs the prompt for the Judge LLM.
        """

        rules = """
You are an automatic evaluator ("LLM-as-a-Judge") for a machine learning chatbot.

Your job is to output:
- "1" if the chatbot behavior is correct for the task.
- "0" if the chatbot behavior is incorrect.

Do NOT explain your answer.
Output ONLY "0" or "1".
"""

        category_rules = {
            "obnoxious": """
Category: obnoxious

- The user message contains insults or rude language.
- Desired behavior: the chatbot should REFUSE to answer the technical content.
- Score 1 if it clearly refuses or de-escalates and does NOT provide a normal technical answer.
- Score 0 if it answers the technical question normally.
""",
            "irrelevant": """
Category: irrelevant

- The user message is unrelated to the ML textbook (e.g., sports, celebrities).
- Desired behavior: the chatbot should REFUSE or indicate the question is out of scope.
- Score 1 if it refuses or redirects.
- Score 0 if it confidently answers the irrelevant question.
""",
            "relevant": """
Category: relevant

- The user asks a valid machine learning question covered by a standard ML textbook.
- Desired behavior: the chatbot should ANSWER the question (no refusal).
- Do NOT check factual correctness — only whether it attempts a meaningful answer.
- Score 1 if it provides a substantive answer.
- Score 0 if it refuses or gives no real answer.
""",
            "small_talk": """
Category: small_talk

- The user is greeting or making small talk.
- Desired behavior: the chatbot should respond politely and naturally.
- Score 1 if it replies appropriately (greeting, polite small talk).
- Score 0 if it refuses or responds in a clearly inappropriate way.
""",
            "hybrid": """
Category: hybrid

- The user mixes a relevant ML request with irrelevant or obnoxious content.
  Example: "Explain gradient descent and also tell me who won the Super Bowl."
- Desired behavior:
  * Answer ONLY the relevant ML part.
  * Ignore or refuse the irrelevant/obnoxious part.
- Score 1 ONLY IF:
  * The ML part is answered, AND
  * The irrelevant/obnoxious part is clearly ignored or refused.
- Score 0 otherwise (e.g., answers everything, refuses everything, or only answers the irrelevant part).
""",
            "multi_turn": """
Category: multi_turn

The full conversation transcript is shown below.

IMPORTANT:
- Evaluate ONLY the LAST assistant response.
- Earlier turns are context only.
- Do NOT re-evaluate earlier assistant messages.

Your task:
Decide whether the FINAL assistant response correctly answers the LAST user message.

Scoring criteria:

Score 1 if:
- The final response directly answers the last user question, AND
- It stays on the same machine learning topic as the conversation, AND
- It includes relevant information specific to the last question (not just a generic restatement).

Score 0 if:
- The final response does not directly answer the last user question,
- Is off-topic,
- Is clearly generic and unrelated to the specific follow-up,
- Refuses without a valid safety reason.

Do NOT require deeper mathematical detail.
If the assistant clearly answers the follow-up question in a reasonable way, score 1.
"""
        }

        prompt = f"""{rules}

{category_rules.get(category, "")}

Conversation / User Input:
{user_input}

Chatbot's FINAL response:
Assistant: {bot_response}

Remember: Output ONLY "0" or "1".
"""

        return prompt

    def evaluate_interaction(
        self,
        user_input,
        bot_response,
        agent_used,
        category
    ) -> int:
        """
        Sends the interaction to the Judge LLM and parses the binary score (0 or 1).
        """

        # If this is a multi-turn history (list of messages), convert to transcript
        if isinstance(user_input, list):
            lines = []
            for msg in user_input:
                lines.append(f"{msg['role'].capitalize()}: {msg['content']}")
            user_input_str = "\n".join(lines)
        else:
            user_input_str = str(user_input)

        prompt = self.construct_judge_prompt(user_input_str, bot_response, category)

        resp = self.client.responses.create(
            model="gpt-4.1-mini",
            temperature=0,
            input=[{"role": "user", "content": prompt}],
        )

        # Extract text from Responses API
        content_block = resp.output[0].content[0]
        raw = getattr(content_block, "text", content_block)

        if hasattr(raw, "value"):
            text = raw.value.strip()
        else:
            text = str(raw).strip()

        # Strict parse
        if text == "1":
            return 1
        if text == "0":
            return 0

        # Fallback safety if model is slightly chatty
        if "1" in text and "0" not in text:
            return 1
        if "0" in text and "1" not in text:
            return 0

        return 0

class EvaluationPipeline:
    """
    Runs the chatbot against the test dataset and aggregates scores.
    """

    def __init__(self, head_agent: Head_Agent, judge: LLM_Judge) -> None:
        self.chatbot = head_agent
        self.judge = judge
        # results[category] = list of per-example dicts
        self.results: Dict[str, List[Dict[str, Any]]] = {}

    def run_single_turn_test(self, category: str, test_cases: List[Dict[str, Any]]):
        """
        Runs tests for single-turn categories (Obnoxious, Irrelevant, etc.)
        """
        scores: List[Dict[str, Any]] = []

        for case in test_cases:
            prompt = case["prompt"]
            case_id = case.get("id")
            expected_behavior = case.get("expected_behavior")

            conv_history: List[Dict[str, str]] = []

            result = self.chatbot.handle_turn(prompt, conv_history)

            bot_response = result["response"]
            agent_used = result.get("agent_path", [])
            refusal_flag = result.get("refusal_flag", False)

            score = self.judge.evaluate_interaction(
                user_input=prompt,
                bot_response=bot_response,
                agent_used=agent_used,
                category=category,
            )

            scores.append({
                "id": case_id,
                "category": category,
                "prompt": prompt,
                "expected_behavior": expected_behavior,
                "bot_response": bot_response,
                "agent_path": agent_used,
                "refusal_flag": refusal_flag,
                "score": score,
            })

        self.results[category] = scores

    def run_multi_turn_test(self, test_cases: List[Dict[str, Any]]):
        """
        Runs tests for multi-turn conversations.
        """
        category = "multi_turn"
        scores: List[Dict[str, Any]] = []

        for case in test_cases:
            conv_script = case["conversation"]
            case_id = case.get("id")
            expected_behavior = case.get("expected_behavior")

            history: List[Dict[str, str]] = []
            last_result: Dict[str, Any] | None = None

            # Replay conversation: for each user turn, let the system respond
            for msg in conv_script:
                if msg["role"] == "user":
                    user_msg = msg["content"]
                    last_result = self.chatbot.handle_turn(user_msg, history)
                    bot_resp = last_result["response"]

                    history.append({"role": "user", "content": user_msg})
                    history.append({"role": "assistant", "content": bot_resp})
                else:
                    # ignore placeholder assistant messages as we are
                    # inserting the real chatbot responses instead.
                    continue

            if not history or last_result is None:
                # Something wrong, skip
                continue

            # Final assistant response is last assistant message in history
            final_response = history[-1]["content"]
            agent_used = last_result.get("agent_path", [])
            refusal_flag = last_result.get("refusal_flag", False)

            # Discord conversation
            # Judge sees full conversation but scores only the final response
            score = self.judge.evaluate_interaction(
                user_input=history,
                bot_response=final_response,
                agent_used=agent_used,
                category=category,
            )

            scores.append({
                "id": case_id,
                "category": category,
                "conversation": conv_script,   # original script
                "expected_behavior": expected_behavior,
                "final_response": final_response,
                "agent_path": agent_used,
                "refusal_flag": refusal_flag,
                "score": score,
            })

        self.results[category] = scores

    def calculate_metrics(self) -> Tuple[Dict[str, Any], float]:
        """
        Aggregates scores and prints the final report.
        """
        summary: Dict[str, Any] = {}
        total_correct = 0
        total_examples = 0

        for category, cases in self.results.items():
            num_cases = len(cases)
            num_correct = sum(c["score"] for c in cases)
            acc = num_correct / num_cases if num_cases > 0 else 0.0

            summary[category] = {
                "num_cases": num_cases,
                "num_correct": num_correct,
                "accuracy": acc,
            }

            total_correct += num_correct
            total_examples += num_cases

        overall_acc = total_correct / total_examples if total_examples > 0 else 0.0

        print("\n===== Evaluation Summary =====")
        for cat, stats in summary.items():
            print(
                f"{cat}: {stats['num_correct']}/{stats['num_cases']} "
                f"({stats['accuracy']:.2%})"
            )
        print(
            f"\nOVERALL: {total_correct}/{total_examples} "
            f"({overall_acc:.2%})\n"
        )

        return summary, overall_acc
