import os

# Use the environment variable if set, otherwise default to localhost
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
print(f"Connecting to Redis at: {REDIS_URL}")
import time

from langchain.globals import set_llm_cache
from langchain.schema import Generation
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_redis import RedisCache, RedisSemanticCache
from utils.LLM import llm,doc_embeddings_model
import langchain_core
import langchain_openai
import openai
import redis


semantic_cache = RedisSemanticCache(
    redis_url=REDIS_URL, embeddings=doc_embeddings_model, distance_threshold=0.01
)
# Set the cache for LangChain to use
set_llm_cache(semantic_cache)
# Function to test semantic cache
def test_semantic_cache(prompt):
    start_time = time.time()
    result = llm.invoke(prompt)
    end_time = time.time()
    return result, end_time - start_time

# Original query
original_prompt = "What is the capital of France?"
result1, time1 = test_semantic_cache(original_prompt)
print(
    f"Original query:\nPrompt: {original_prompt}\nResult: {result1}\nTime: {time1:.2f} seconds\n"
)

# Semantically similar query
similar_prompt = "who i am "
result2, time2 = test_semantic_cache(similar_prompt)
print(
    f"Similar query:\nPrompt: {similar_prompt}\nResult: {result2}\nTime: {time2:.2f} seconds\n"
)

print(f"Speed improvement: {time1 / time2:.2f}x faster")

# Clear the semantic cache
semantic_cache.clear()
print("Semantic cache cleared")