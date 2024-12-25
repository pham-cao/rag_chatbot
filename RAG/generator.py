from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from RAG.LLMs import llm, doc_embeddings_model
from langchain_qdrant import QdrantVectorStore
from utils.vector_db import client_db
from langchain.chains import create_history_aware_retriever

vector_store = QdrantVectorStore(
    client=client_db,
    collection_name="Nội quy công ty GEM 2.docx",
    embedding=doc_embeddings_model,
)
retriever = vector_store.as_retriever()
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
rag_chain = contextualize_q_prompt|llm
from langchain_core.messages import HumanMessage

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

from langchain_core.messages import HumanMessage

chat_history = []

question = "mức lương 30tr 1 tháng làm 9 tháng ở gem được bao nhiêu tiền ?"
ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
chat_history.extend([HumanMessage(content=question), ai_msg_1["answer"]])

second_question = "đủ một năm thì sao ?"
ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})

print(ai_msg_2["answer"])