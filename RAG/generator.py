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
from langchain_core.messages import HumanMessage, AIMessage
from redisvl.extensions.llmcache import SemanticCache
from PROMPT import *
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)



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
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain



from redisvl.utils.vectorize import CustomTextVectorizer

vectorizer = CustomTextVectorizer(embed=doc_embeddings_model.embed_query)



class RAGChain:
    def __init__(self):
        self.chat_history = []
        self.retriever = init_retriever()
        self.history_aware_retriever = init_history_aware_retriever(self.retriever)
        self.rag_chain = init_rag_chain(self.history_aware_retriever)

        self.cache = SemanticCache(
            name="llmcache",  # underlying search index name
            prefix="llmcache",  # redis key prefix for hash entries
            redis_url="redis://localhost:6379",  # redis connection url string
            distance_threshold=0.01,
            vectorizer=vectorizer# semantic cache distance threshold,

        )
        self.cache.clear()

        self.rewrite_prompt = PromptTemplate(template=REWRITE_PROMPT,
                                             input_variables=["query", "response"]
                                             )

        self.rewrite_chain = self.rewrite_prompt | llm

    def invoke(self, question):
        if response := self.cache.check(prompt=question, return_fields=["response"]):
            response = response[0]["response"]
            response = self.rewrite_chain.invoke({"query": question, "response": response})


        else:

            response = self.rag_chain.invoke({"input": question, "chat_history": self.chat_history})["answer"]
            self.cache.store(
                prompt=question,
                response=response
            )
        self.chat_history.extend([HumanMessage(content=question), AIMessage(content=response)])
        return response
