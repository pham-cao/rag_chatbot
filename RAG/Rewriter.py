from langchain_core.messages import HumanMessage, AIMessage
from RAG.LLMs import llm
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from RAG.PROMPT import *



class ReWriter:
    def __init__(self):
        self.chat_history = []
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUALIZE_PROMPT),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.llm = self.prompt|llm
    def invoke(self, question):
        response = self.llm.invoke({"input": question, "chat_history": self.chat_history})
        return response

    def add_history(self, question,response):
        if len(self.chat_history) > 5:
            self.chat_history.pop(0)
        self.chat_history.extend([HumanMessage(content=question), AIMessage(content=response)])

