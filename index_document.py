from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.node_parser.topic import TopicNodeParser


text = """In this paper, we introduce a novel graph RAG method for applying LLMs to the medical domain, which we refer to as Medical Graph RAG (MedRAG). This technique improves LLM performance in the medical domain by response queries with grounded source citations and clear interpretations of medical terminology, boosting the transparency and interpretability of the results. This approach involves a three-tier hierarchical graph construction method. Initially, we use documents provided by users as our top-level source to extract entities. These entities are then linked to a second level consisting of more basic entities previously abstracted from credible medical books and papers. Subsequently, these entities are connected to a third level—the fundamental medical dictionary graph—that provides detailed explanations of each medical term and their semantic relationships. We then construct a comprehensive graph at the highest level by linking entities based on their content and hierarchical connections. This method ensures that the knowledge can be traced back to its sources and the results are factually accurate.

To respond to user queries, we implement a U-retrieve strategy that combines top-down retrieval with bottom-up response generation. The process begins by structuring the query using predefined medical tags and indexing them through the graphs in a top-down manner. The system then generates responses based on these queries, pulling from meta-graphs—nodes retrieved along with their TopK related nodes and relationships—and summarizing the information into a detailed response. This technique maintains a balance between global context awareness and the contextual limitations inherent in LLMs.

Our medical graph RAG provides Intrinsic source citation can enhance LLM transparency, interpretability, and verifiability. The results provides the provenance, or source grounding information, as it generates each response, and demonstrates that an answer is grounded in the dataset. Having the cited source for each assertion readily available also enables a human user to quickly and accurately audit the LLM’s output directly against the original source material. It is super useful in the field of medicine that security is very important, and each of the reasoning should be evidence-based. By using such a method, we construct an evidence-based Medical LLM that the clinician could easiely check the source of the reasoning and calibrate the model response to ensure the safty usage of llm in the clinical senarios.

To evaluate our medical graph RAG, we implemented the method on several popular open and closed-source LLMs, including ChatGPT OpenAI (2023a) and LLaMA Touvron et al. (2023), testing them across mainstream medical Q&A benchmarks such as PubMedQA Jin et al. (2019), MedMCQA Pal et al. (2022), and USMLE Kung et al. (2023). For the RAG process, we supplied a comprehensive medical dictionary as the foundational knowledge layer, the UMLS medical knowledge graph Lindberg et al. (1993) as the foundamental layer detailing semantic relationships, and a curated MedC-K dataset Wu et al. (2023) —comprising the latest medical papers and books—as the intermediate level of data to simulate user-provided private data. Our experiments demonstrate that our model significantly enhances the performance of general-purpose LLMs on medical questions. Remarkably, it even surpasses many fine-tuned or specially trained LLMs on medical corpora, solely using the RAG approach without additional training.
"""

from llama_index.core import Document

documents = [Document(text=text)]

embed_model = OpenAIEmbedding()
llm = OpenAI(model="gpt-4o-mini")

reader = SimpleDirectoryReader(input_files=['thachsanhlythong.pdf'])
document = reader.load_data()

print(document[0].text)

node_parser = TopicNodeParser.from_defaults(
    llm=llm,
    max_chunk_size=1000,
    similarity_method="llm",  # can be "llm" or "embedding"
    window_size=2,  # paper suggests window_size=5
)
nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)