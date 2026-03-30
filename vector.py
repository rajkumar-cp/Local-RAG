import os

import pandas
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

file = pandas.read_csv("realistic_restaurant_reviews.csv")
embeddingModel = OllamaEmbeddings(model="mxbai-embed-large")
vector_db_location = "./vector-db"
documents = []
ids = []
add_docs = not os.path.exists(vector_db_location)
print(add_docs)
if add_docs:
    os.makedirs(vector_db_location)
    for index, row in file.iterrows():
        document = Document(
            page_content=row["Title"]+row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(index)
        )
        documents.append(document)
        ids.append(str(index))
print(f"Documents added: {len(documents)}")
vector_store = Chroma(
    collection_name="restaurant_reviews",
    embedding_function=embeddingModel,
    persist_directory=vector_db_location
)

if add_docs:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k":5})