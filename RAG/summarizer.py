from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from RAG.LLMs import llm
from langchain_core.prompts import PromptTemplate

def load_summarizer_chain(docs):

    prompt_template = """Vui lòng tóm tắt đoạn văn dưới đây, giữ lại các ý chính và thông tin quan trọng. 
    Loại bỏ các chi tiết không cần thiết, nhưng vẫn đảm bảo thông tin cốt lõi được truyền đạt:
    "{text}"
    """
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    return stuff_chain.run(docs)