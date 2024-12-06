from LLM import llm, query_embeddings_model, doc_embeddings_model
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from qdrant_client import QdrantClient
from decouple import config
from qdrant_client.http.models import Distance, VectorParams
from uuid import uuid4
from langchain_qdrant import RetrievalMode

from langchain_qdrant import QdrantVectorStore

QDRANT_HOST = config('QDRANT_HOST')
QDRANT_POST = config('QDRANT_PORT')

client_db = QdrantClient(host=QDRANT_HOST,
                         port=QDRANT_POST)


def load_docs_from_text(documents: str, book_name: str):
    client_db.create_collection(
        collection_name=book_name,
        vectors_config=VectorParams(size=768,
                                    distance=Distance.COSINE),
    )
    vector_store = QdrantVectorStore(
        client=client_db,
        collection_name=book_name,
        embedding=doc_embeddings_model,
    )

    # chunking document

    semantic_chunker = SemanticChunker(embeddings=doc_embeddings_model,
                                       breakpoint_threshold_type="percentile")

    semantic_chunks = semantic_chunker.create_documents([documents])
    uuids = [str(uuid4()) for _ in range(len(semantic_chunks))]
    vector_store.add_documents(documents=semantic_chunks, ids=uuids)

def search_query(collection_name, query):
    vector_store = QdrantVectorStore(
        client=client_db,
        collection_name=collection_name,
        embedding=doc_embeddings_model,
    )
    results  = vector_store.similarity_search(query,k=2)
    return results
