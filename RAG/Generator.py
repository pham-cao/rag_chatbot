from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from RAG.LLMs import llm, doc_embeddings_model
from langchain_qdrant import QdrantVectorStore
from utils.vector_db import client_db
from langchain.chains import create_history_aware_retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
import logging
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from RAG.PROMPT import *
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


from RAG.Rewriter import ReWriter
from RAG.SematicCache import CachingSearch


vector_store = QdrantVectorStore(
    client=client_db,
    collection_name="Nội quy công ty GEM 2.docx",
    embedding=doc_embeddings_model,
)


class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""

    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines


def init_retriever():
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template=MUTIL_QUERY_PROMPT,
    )

    llm_chain = QUERY_PROMPT | llm | LineListOutputParser()

    # Run
    retriever = MultiQueryRetriever(
        retriever=vector_store.as_retriever(), llm_chain=llm_chain, parser_key="lines"
    )
    return retriever


def init_history_aware_retriever(retriever):
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


def init_rag_chain(history_aware_retriever):
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            # MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


class RAGChain:
    def __init__(self):
        self.rewriter = ReWriter()
        self.cache = CachingSearch()
        # self.cache.cache.clear()
        self.retriever = init_retriever()
        self.rag_chain = init_rag_chain(self.retriever)

    def invoke(self, question):
        question_rewrite = self.rewriter.invoke(question)
        print(question_rewrite)

        response = self.cache.search(question_rewrite)
        if response is None:
            response = self.rag_chain.invoke({"input": question_rewrite })["answer"]
        self.cache.add_cache(question_rewrite,response)
        self.rewriter.add_history(question_rewrite,response)
        return response

