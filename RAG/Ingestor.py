from langchain.text_splitter import RecursiveCharacterTextSplitter
from RAG.Summarizer import load_summarizer_chain
from RAG.LLMs import llm, query_embeddings_model
from langchain.schema.document import Document
from utils.vector_db import client_db, create_new_collection
from more_itertools import chunked
from uuid import uuid4
from langchain_qdrant import QdrantVectorStore


from langchain_experimental.text_splitter import SemanticChunker
#
# SemanticChunker(embeddings=doc_embeddings_model,
#                                        breakpoint_threshold_type="percentile")


class Ingestor:
    def __init__(self):
        self.db_client = client_db
        self.embedding_model = query_embeddings_model
        # self.splitter = RecursiveCharacterTextSplitter(chunk_size=1028, chunk_overlap=256)

        self.splitter = SemanticChunker(embeddings=query_embeddings_model,
                                       breakpoint_threshold_type="percentile")
        self.summarizer = load_summarizer_chain

    def insert(self, text: str, collection_name: str):
        # create collection
        summarizer_docs = self.summarizer([Document(page_content=text)])
        create_new_collection(collection_name, summarizer_docs)

        chunks = self.splitter.create_documents([text])

        chunks_split = chunked(chunks, 5)
        for Paragraph  in chunks_split:
            summarize_content = "**tóm tắt**"+self.summarizer(Paragraph)
            chunks.append(Document(page_content=summarize_content))
        uuids = [str(uuid4()) for _ in range(len(chunks))]

        vector_store = QdrantVectorStore(
            client=client_db,
            collection_name=collection_name,
            embedding=self.embedding_model,
        )

        vector_store.add_documents(documents=chunks, ids=uuids)

