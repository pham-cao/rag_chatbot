from redisvl.extensions.llmcache import SemanticCache
from redisvl.utils.vectorize import CustomTextVectorizer
from RAG.LLMs import doc_embeddings_model,llm
from langchain.prompts import PromptTemplate
from RAG.PROMPT import *

vectorizer = CustomTextVectorizer(embed=doc_embeddings_model.embed_query)


class CachingSearch:
    def __init__(self):
        self.cache = SemanticCache(
            name="llmcache",  # underlying search index name
            prefix="llmcache",  # redis key prefix for hash entries
            redis_url="redis://localhost:6379",  # redis connection url string
            distance_threshold=0.01,
            vectorizer=vectorizer  # semantic cache distance threshold,

        )
        self.rewrite_prompt = PromptTemplate(template=REWRITE_PROMPT,
                                             input_variables=["query", "response"]
                                             )

        self.rewrite_chain = self.rewrite_prompt | llm
    def search(self, question):
        if response := self.cache.check(prompt=question, return_fields=["response"]):
            response = response[0]["response"]
            response = self.rewrite_chain.invoke({"query": question, "response": response})
            return response
        else:

            print("Empty cache")
            return None
    def add_cache(self,question,response):
        self.cache.store(
            prompt=question,
            response=response
        )
