from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional,Literal
import google.generativeai as genai
import os
from  decouple import config

GOOGLE_API_KEY = config('GOOGLE_API_KEY')

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasources: List[Literal["python_docs", "js_docs", "golang_docs"]] = Field(
        ...,
        description="Given a user question choose which datasources would be most relevant for answering their question",
    )


class QueryRouter:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
        )

        self.parser = PydanticOutputParser(pydantic_object=RouteQuery)


        self.prompt_template = PromptTemplate(
            template="""Given the query: "{query}", analyze its intent, context, and domain to determine the most 
            relevant data source from the following options:
            {format_instructions}

            Your task is to select the single most relevant source or specify if multiple sources are needed. 
            Provide a justification for your choice and summarize the key criteria used in your decision.

            
            """,
            input_variables=["query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )

    def route_query(self, query: str):
        # Generate the prompt
        prompt = self.prompt_template.format(query=query)

        # Get response from Gemini
        response = self.llm.invoke(prompt)

        # Parse the response
        parsed_response = self.parser.parse(response.content)

        print(parsed_response)

if __name__ == "__main__":
    # Initialize the router
    router = QueryRouter(api_key="YOUR_GOOGLE_API_KEY")
    query = "is there feature parity between the Python and JS implementations of OpenAI chat models"
    router.route_query(query)

