# part3/agents.py
import re
import json

from .prompts import (
    OBNOXIOUS_PROMPT,
    DOMAIN_PROMPT,
    REWRITER_PROMPT,
    RELEVANCE_PROMPT,
    ANSWERING_PROMPT,
)

class Obnoxious_Agent:
    def __init__(self, client) -> None:
        self.client = client
        self.model = "gpt-4.1-nano"
        self.prompt = OBNOXIOUS_PROMPT 

    def set_prompt(self, prompt):
        self.prompt = prompt

    def extract_action(self, response) -> bool:
        txt = (response or "").strip().lower()

        # minor formatting, just in case, cant trust AI model response Lol
        first = re.split(r"\s+", txt)[0] if txt else ""
        if first == "yes":
            return True
        if first == "no":
            return False

        # if the model misbehaves, refuse
        return True

    def check_query(self, query):
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": query},
            ],
            temperature=0,
        )
        out = (r.choices[0].message.content or "").strip()
        return self.extract_action(out)


class Context_Rewriter_Agent:
    def __init__(self, openai_client):
        self.client = openai_client
        self.model = "gpt-4.1-nano"
        self.prompt = REWRITER_PROMPT

    def rephrase(self, user_history, latest_query):
        tail = user_history[-6:] if user_history else []

        convo = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in tail
        )

        r = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt},
                {
                    "role": "user",
                    "content": (
                        "CONVERSATION HISTORY:\n"
                        f"{convo}\n\n"
                        "LATEST USER MESSAGE:\n"
                        f"{latest_query}\n\n"
                        "Rewrite the latest user message as a standalone query:"
                    ),
                },
            ],
            temperature=0,
        )

        rewritten = (r.choices[0].message.content or latest_query).strip()
        return rewritten


class Query_Agent:
    def __init__(self, pinecone_index, openai_client, embeddings) -> None:
        self.index = pinecone_index
        self.client = openai_client
        self.embeddings = embeddings

        self.model = "gpt-4.1-nano"

        self.prompt = DOMAIN_PROMPT

    def query_vector_store(self, query, k=5):
        results = self.index.similarity_search_with_score(query, k=k)

        docs = []

        for doc, score in results:
            md = doc.metadata or {}
            docs.append({
                "id": f"page-{int(md.get('page_number', -1))}",
                "score": float(score),
                "text": doc.page_content,
                "page_number": md.get("page_number"),
                "num_tokens": md.get("num_tokens"),
            })

        return docs

    def set_prompt(self, prompt):
        self.prompt = prompt

    def extract_action(self, query=None):
        if not query:
            return False

        r = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": query},
            ],
            temperature=0,
        )

        out = (r.choices[0].message.content or "").strip().lower()
        first = out.split()[0] if out else ""

        if first == "yes":
            return True
        if first == "no":
            return False

        # if model misbehaves
        return False

    def run(self, query, k=5):
        in_domain = self.extract_action(query=query)
        docs = self.query_vector_store(query, k=k) if in_domain else []
        return {"in_domain": in_domain, "docs": docs}

class Answering_Agent:
    def __init__(self, openai_client) -> None:
        self.client = openai_client
        self.prompt = ANSWERING_PROMPT

    def generate_response(self, query, docs, conv_history, k=5):
        context_parts = []
        total = 0
        max_chars = 6000

        for d in docs[:k]:
            block = f"PAGE {d.get('page_number')}:\n{(d.get('text') or '').strip()}"
            if total + len(block) > max_chars:
                break

            context_parts.append(block)
            total += len(block)

        context = "\n\n".join(context_parts)

        r = self.client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": self.prompt},
                {
                    "role": "user",
                    "content": (
                        "Use the CONTEXT below to answer the QUERY.\n"
                        "Remember: answer only what is supported by the CONTEXT; "
                        "ignore other parts of the query.\n\n"
                        f"QUERY:\n{query}\n\nCONTEXT:\n{context}"
                    ),
                },
            ],
            temperature=0,
        )

        # Despite strict prompt constraints, gpt-4.1-nano may still
        # produce meta-statements about missing context in partial-answer cases.
        # To guarantee compliance with the assignment rules,
        # we enforce a deterministic post-processing filter that removes
        # sentences referencing missing or unavailable context.
        raw = (r.choices[0].message.content or "").strip()

        sentences = re.split(r'(?<=[.!?])\s+', raw)
        banned_fragments = [
            "not mentioned in the context",
            "not mentioned in the provided context",
            "not mentioned in the provided information",
            "not mentioned in the information",
            "not present in the context",
            "not present in the provided information",
            "not covered in the context",
            "not covered in the provided information",
        ]

        filtered = [
            s for s in sentences
            if not any(b in s.lower() for b in banned_fragments)
        ]

        answer = " ".join(filtered).strip()

        # fall back to raw if answer is nothing
        return answer or raw

class Relevant_Documents_Agent:
    def __init__(self, openai_client) -> None:
        self.client = openai_client
        self.model = "gpt-4.1-nano"
        self.prompt = RELEVANCE_PROMPT

    def get_relevance(self, conversation) -> str:
        r = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": conversation},
            ],
            temperature=0,
        )
        return (r.choices[0].message.content or "").strip()

    def judge_docs(self, query, docs):
        doc_blob = []
        for d in docs[:5]:
            text = (d.get("text") or "").strip()
            doc_blob.append(
                f"[PAGE {d.get('page_number')} score={d.get('score')}]\n{text}"
            )

        conversation = f"USER QUERY:\n{query}\n\nRETRIEVED DOCS:\n" + "\n\n".join(doc_blob)
        raw = self.get_relevance(conversation)

        try:
            data = json.loads(raw)
            return bool(data["relevant"]), str(data["reason"])
        except Exception:
            return False, "Could not parse relevance JSON."
