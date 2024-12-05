from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.node_parser.topic import TopicNodeParser
from llama_index.llms.gemini import Gemini
from decouple import config
from llama_index.embeddings.gemini import GeminiEmbedding

GOOGLE_API_KEY = 'AIzaSyBIi_2tBNl0F9SqvuFenZ4Sla3pDv3eI2Q'
llm = Gemini(model="models/gemini-pro",
             api_key=GOOGLE_API_KEY)
embed_model = GeminiEmbedding(
    model_name="models/embedding-001", api_key=GOOGLE_API_KEY, title="this is a document"
)

# from llama_index.core import Document
#
# documents = [Document(text=text)]

# embed_model = OpenAIEmbedding()

reader = SimpleDirectoryReader(input_files=['thachsanhlythong.pdf'])
document = reader.load_data()

print(document[0].text)

node_parser = TopicNodeParser.from_defaults(
    embed_model=embed_model,
    llm=llm,
    max_chunk_size=300,
    similarity_method="embedding",  # can be "llm" or "embedding"
    window_size=2,  # paper suggests window_size=5
)
nodes = node_parser.get_nodes_from_documents(document, show_progress=True)
print(nodes[0].get_content())
