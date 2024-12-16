from transformers import AutoTokenizer, AutoModel
import torch

# Tải tokenizer và model
tokenizer = AutoTokenizer.from_pretrained("keepitreal/vietnamese-sbert")
model = AutoModel.from_pretrained("keepitreal/vietnamese-sbert")

# Encode văn bản
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)

# Lấy embeddings từ đầu ra
sentence_embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
print(sentence_embeddings)
