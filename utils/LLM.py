from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from decouple import config

GOOGLE_API_KEY = config('GOOGLE_API_KEY')
model_generative_name = "gemini-pro"
model_embeddings_name = "models/embedding-001"

# init model using RAG
llm = GoogleGenerativeAI(model=model_generative_name,
                         google_api_key=GOOGLE_API_KEY)

query_embeddings_model = GoogleGenerativeAIEmbeddings(model=model_embeddings_name,
                                                      task_type="retrieval_query",
                                                      google_api_key=GOOGLE_API_KEY)

doc_embeddings_model = GoogleGenerativeAIEmbeddings(model=model_embeddings_name,
                                                    task_type="retrieval_document",
                                                    google_api_key=GOOGLE_API_KEY)
