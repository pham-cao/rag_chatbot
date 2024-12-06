from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
import google.generativeai as genai
import os

from decouple import config
GOOGLE_API_KEY = config('GOOGLE_API_KEY')


# Define the function schemas
class FunctionCall(BaseModel):
    function_name: str = Field(description="Name of the function to call")
    reason: str = Field(description="Reason for choosing this function")
    parameters: dict = Field(description="Parameters to pass to the function")


class RouterResponse(BaseModel):
    function_calls: List[FunctionCall] = Field(description="List of function calls to make")


# Example functions that could be called
def search_products(query: str, category: Optional[str] = None):
    return f"Searching for products: {query} in category: {category}"


def get_order_status(order_id: str):
    return f"Getting status for order: {order_id}"


def create_support_ticket(issue: str, priority: str):
    return f"Creating support ticket: {issue} with priority: {priority}"


class QueryRouter:
    def __init__(self, api_key: str):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0,
        )

        self.parser = PydanticOutputParser(pydantic_object=RouterResponse)

        self.function_registry = {
            "search_products": search_products,
            "get_order_status": get_order_status,
            "create_support_ticket": create_support_ticket
        }

        self.prompt_template = PromptTemplate(
            template="""Analyze the following query and determine which function(s) should be called.
            Available functions:

            1. search_products(query: str, category: Optional[str])
               - Searches for products matching the query

            2. get_order_status(order_id: str)
               - Retrieves the status of an order

            3. create_support_ticket(issue: str, priority: str)
               - Creates a support ticket

            Query: {query}

            {format_instructions}
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

        results = []
        # Execute the function calls
        for function_call in parsed_response.function_calls:
            if function_call.function_name in self.function_registry:
                func = self.function_registry[function_call.function_name]
                result = func(**function_call.parameters)
                results.append({
                    "function": function_call.function_name,
                    "reason": function_call.reason,
                    "result": result
                })

        return results


# Example usage
if __name__ == "__main__":
    # Initialize the router
    router = QueryRouter(api_key="YOUR_GOOGLE_API_KEY")

    # Test queries
    test_queries = [
        "I'm looking for red shoes in the footwear category",
        "What's the status of my order #12345?",
        "I need help, my account is locked and it's urgent"
    ]

    # Process each query
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = router.route_query(query)
        for result in results:
            print(f"\nFunction: {result['function']}")
            print(f"Reason: {result['reason']}")
            print(f"Result: {result['result']}")
