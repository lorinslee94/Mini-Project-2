# part3/head_agent.py

from .agents import (
    Obnoxious_Agent,
    Context_Rewriter_Agent,
    Query_Agent,
    Answering_Agent,
    Relevant_Documents_Agent,
)
from .eval_schema import REFUSAL_GENERAL, REFUSAL_NO_CONTEXT, make_result, is_refusal

from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

class Head_Agent:
    def __init__(self, openai_key, pinecone_key, pinecone_index_name) -> None:
        self.openai_key = openai_key
        self.pinecone_key = pinecone_key
        self.pinecone_index_name = pinecone_index_name

        # OpenAI Client (for all agents)
        self.openai_client = OpenAI(api_key=openai_key)

        # Embeddings (for retrieval)
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=openai_key,
        )

        # Pinecone Vector Store
        self.namespace = "ns1000"
        self.vectorstore = PineconeVectorStore.from_existing_index(
            index_name=pinecone_index_name,  # ml-textbook-rag-1536
            embedding=self.embeddings,
            text_key="text",
            namespace=self.namespace,
        )
        self.pinecone_index = self.vectorstore

        self.setup_sub_agents()

    def setup_sub_agents(self):
        self.obnoxious = Obnoxious_Agent(self.openai_client)
        self.rewriter = Context_Rewriter_Agent(self.openai_client)
        self.query_agent = Query_Agent(self.pinecone_index, self.openai_client, self.embeddings)
        self.relevance_agent = Relevant_Documents_Agent(self.openai_client)
        self.answering_agent = Answering_Agent(self.openai_client)

    def is_small_talk(self, query):
        q = query.lower().strip()

        if len(q.split()) > 6:
            return False

        small_talk_phrases = [
            "hello", "hi", "hey",
            "good morning", "good afternoon", "good evening",
            "how are you", "what's up"
        ]

        return any(phrase in q for phrase in small_talk_phrases)

    # Helper function to be called in the Streamlit app
    def handle_turn(self, user_query, conv_history):
        agent_path = ["Head_Agent"]

        agent_path.append("Obnoxious_Agent")
        if self.obnoxious.check_query(user_query):
            return make_result(REFUSAL_GENERAL, agent_path, True)

        # Small talk handling (before Query_Agent)
        if self.is_small_talk(user_query):
            agent_path.append("SmallTalk_Handler")
            return make_result(
                "Hello! I’m here to help with machine learning questions. How can I assist you today?",
                agent_path,
                False
            )

        agent_path.append("Context_Rewriter_Agent")
        rewritten = self.rewriter.rephrase(conv_history, user_query)

        agent_path.append("Query_Agent")
        qres = self.query_agent.run(rewritten, k=5)
        if not qres["in_domain"]:
            return make_result(REFUSAL_GENERAL, agent_path, True)

        docs = qres["docs"]

        agent_path.append("Relevant_Documents_Agent")
        relevant, _reason = self.relevance_agent.judge_docs(rewritten, docs)
        if not relevant:
            return make_result(REFUSAL_NO_CONTEXT, agent_path, True)

        agent_path.append("Answering_Agent")

        answer = self.answering_agent.generate_response(
            rewritten, docs, conv_history, k=5)

        return make_result(
            answer, 
            agent_path, 
            is_refusal(answer),
            doc_ids=[d["id"] for d in docs],
            scores=[d["score"] for d in docs]
        )
