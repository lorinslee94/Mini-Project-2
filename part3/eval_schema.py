# part3/eval_schema.py

REFUSAL_GENERAL = "REFUSAL: I can’t help with that request."
REFUSAL_NO_CONTEXT = "REFUSAL: I don’t have enough relevant context in the indexed book to answer that."

def make_result(response, agent_path, refusal_flag, **metadata):
    result = {
        "response": response,
        "agent_path": agent_path,
        "refusal_flag": refusal_flag,
    }

    if metadata:
        result["metadata"] = metadata

    return result

def is_refusal(text):
    return (text or "").strip().startswith("REFUSAL:")

