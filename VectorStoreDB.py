from utils.LLM import doc_embeddings_model
from langchain_experimental.text_splitter import SemanticChunker
from uuid import uuid4
from langchain_qdrant import QdrantVectorStore
from utils.vector_db import client_db,create_new_collection


def load_docs_from_text(documents: str, collection_name: str,description :str):

    create_new_collection(collection_name,description)

    vector_store = QdrantVectorStore(
        client=client_db,
        collection_name=collection_name,
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
    results = vector_store.similarity_search(query, k=2)
    return results
