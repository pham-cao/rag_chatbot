from redisvl.utils.vectorize import CustomTextVectorizer

vectorizer = CustomTextVectorizer(embed=doc_embeddings_model.embed_query)
embedding = vectorizer.embed("Hello, world!")
print(embedding)