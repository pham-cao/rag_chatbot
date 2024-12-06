from langchain_community.document_loaders import PyPDFLoader
from VectorStoreDB import load_docs_from_text,search_query
#
file_name = 'thachsanhlythong.pdf'
# loader = PyPDFLoader(file_name)
# docs = loader.load()
# str = ''
# for doc in docs:
#     str += doc.page_content
#
# load_docs_from_text(str,file_name)

query = ' thạch sach cưới ai '
results = search_query(file_name,query)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")


