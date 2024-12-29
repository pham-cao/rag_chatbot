### Hệ thống RAG Question Answer theo Tài Liệu 
#### Sơ đồ hệ thống
![RAG Architecture Flow as](https://github.com/pham-cao/rag_chatbot/blob/main/images/architecture.png?raw=true)

### Requirement

1. Install Qdrant Database and Redis Database:
``` commandline
docker run -d --name redis-stack-server -p 6379:6379 redis/redis-stack-server:latest

docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```
2. Install package

```commandline
pip install -r requirement.txt
```
### RUN
```commandline
streamlit run "Assistant Chat.py"
```