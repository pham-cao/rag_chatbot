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

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# MUTIL_QUERY_PROMPT = """You are an AI language model assistant. Your task is to generate five
#     different versions of the given user question to retrieve relevant documents from a vector
#     database. By generating multiple perspectives on the user question, your goal is to help
#     the user overcome some of the limitations of the distance-based similarity search.
#     Provide these alternative questions separated by newlines.
#     Original question: {question}"""
MUTIL_QUERY_PROMPT = """Bạn là một trợ lý mô hình ngôn ngữ AI. Nhiệm vụ của bạn là tạo ra năm phiên bản khác nhau của câu hỏi 
người dùng đã cho để truy xuất các tài liệu liên quan từ cơ sở dữ liệu vector. 
Bằng cách tạo ra nhiều góc nhìn khác nhau về câu hỏi của người dùng, mục tiêu của bạn là giúp người dùng vượt qua 
một số hạn chế của việc tìm kiếm dựa trên sự tương đồng theo khoảng cách.
Cung cấp những câu hỏi thay thế này, mỗi câu cách nhau bằng một dòng mới.
Câu hỏi gốc: {question}"""

# CONTEXTUALIZE_PROMPT = """Given a chat history and the latest user question \
# which might reference context in the chat history, formulate a standalone question \
# which can be understood without the chat history. Do NOT answer the question, \
# just reformulate it if needed and otherwise return it as is."""
CONTEXTUALIZE_PROMPT = """ Dựa trên lịch sử trò chuyện và câu hỏi người dùng mới nhất có thể tham chiếu đến ngữ cảnh trong lịch sử trò chuyện, hãy xây dựng lại câu hỏi độc lập sao cho có thể hiểu được mà không cần đến lịch sử trò chuyện. ĐỪNG trả lời câu hỏi, chỉ cần tái cấu trúc câu hỏi nếu cần và nếu không cần thay đổi thì trả lại câu hỏi nguyên bản."""
# QA_SYSTEM_PROMPT = """You are an assistant for question-answering tasks. \
# Use the following pieces of retrieved context to answer the question. \
# If you don't know the answer, just say that you don't know. \
# Use three sentences maximum and keep the answer concise.\
#
# {context}"""
QA_SYSTEM_PROMPT = """
Bạn là trợ lý cho các nhiệm vụ trả lời câu hỏi. Hãy sử dụng các đoạn ngữ cảnh đã được truy xuất dưới đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết. Hãy giữ câu trả lời ngắn gọn và không quá ba câu.

{context}"""
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

class RAGChain:
    def __init__(self):
        self.chat_history = []
        self.retriever = init_retriever()
        self.history_aware_retriever = init_history_aware_retriever(self.retriever)
        self.rag_chain = init_rag_chain(self.history_aware_retriever)
    def invoke(self,question):
        ai_msg = self.rag_chain.invoke({"input": question, "chat_history": self.chat_history})
        self.chat_history.extend([HumanMessage(content=question), AIMessage(content=ai_msg["answer"])])
        return ai_msg["answer"]

