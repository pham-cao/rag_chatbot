from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import google.generativeai as genai
import os
from decouple import config

from utils.vector_db import get_list_collection_names

GOOGLE_API_KEY = config('GOOGLE_API_KEY')


class Document(BaseModel):
    id: str = Field(description="id of Document")
    document_name: str = Field(description="Name of Document")
    description_document: str = Field(description="description of Document")


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasources: List[Document] = Field(
        ...,
        description="Given a user question choose which datasources would be most relevant for answering their question",
    )


class QueryRouter:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
        )

        self.parser = PydanticOutputParser(pydantic_object=RouteQuery)
        collection_names = [str(i) for i in get_list_collection_names()]



        self.prompt_template = PromptTemplate(
            template="""Given the query: "{query}", analyze its intent, context, and domain to determine the most 
            relevant data source from the following options:
            {sources}
      

             Your task is to select the single most relevant source from description of document. 
            Provide a justification for your choice and summarize the key criteria used in your decision.
            
            {format_instructions}

            
            """,
            input_variables=["query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions(),
                               'sources': ''.join(collection_names)}
        )

    def route_query(self, query: str):

        import time
        time1 = time.time()
        # Generate the prompt
        prompt = self.prompt_template.format(query=query)

        print("time temp:", time.time() - time1)



        # Get response from Gemini
        response = self.llm.invoke(prompt)
        print("time LLM response:", time.time() - time1)

        # Parse the response
        parsed_response = self.parser.parse(response.content)



        return parsed_response.datasources[0].document_name
