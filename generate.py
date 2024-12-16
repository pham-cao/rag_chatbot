from utils.LLM import llm, query_embeddings_model
from langchain_qdrant import QdrantVectorStore
from utils.vector_db import client_db
from langchain_core.prompts import PromptTemplate
from RouterQuery import QueryRouter
import time

query_router = QueryRouter()


def generate_answer(question):

    # question = "GEM vao lam viec luc may gio"

    time1 = time.time()
    collection_name = query_router.route_query(question)
    time2 = time.time()
    print(collection_name)
    vector_store = QdrantVectorStore(client=client_db,
                                     collection_name=collection_name,
                                     embedding=query_embeddings_model)

    vector_store_rtv = vector_store.as_retriever()

    result = vector_store_rtv.invoke(question)
    y = " ".join([x.page_content for x in result])
    time3 = time.time()

    prompt = PromptTemplate.from_template("""      
        Bạn Là Elsa , một nữ nhân viên tư vấn thông minh , có tài ăn nói và  có nhiều năm kinh nghiệm trong việc tư vấn và hỗ trợ khách hàng của GEM .
        các thông tin cung cấp dưới ngữ cảnh là các thông tin của công ty GEM .
        Bạn thực hiện bước sau:Đọc ngữ cảnh bên dưới và hiểu tất cả nội dung trong đó 
            Ngữ cảnh: {matching_engine_response}
         sau đó suy luận từ bước rồi trả lời câu hỏi của người dùng một cách chi tiết, đầy đủ và tự nhiên.
        câu hỏi của người dùng: {question}
        
        Lưu ý quan trọng:
        - Chỉ đưa ra câu trả lời cho câu hỏi, không trả về bất kỳ thông tin nào ngoài câu trả lời.
        - Nếu ngữ cảnh không có thông tin liên quan , trả lời: "Không tìm thấy thông tin này."
        - Nếu các thông tin từ câu hỏi không liên quan đến GEM hãy trả lời không hỗ trợ 
                """)

    chain = prompt | llm
    anwser = chain.invoke({"matching_engine_response": y,
                           "question": question})
    time4 = time.time()
    print(time2 - time1)
    print(time3 - time2)
    print(time4 - time3)
    return anwser
