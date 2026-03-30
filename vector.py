import os

import pandas
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

file = pandas.read_csv("realistic_restaurant_reviews.csv")
embeddingModel = OllamaEmbeddings(model="mxbai-embed-large")
vector_db_location = "./vector-db"

vector_store = Chroma(
    collection_name="restaurant_reviews",
    embedding_function=embeddingModel,
    persist_directory=vector_db_location
)

if not os.path.exists(vector_db_location):
    os.makedirs(vector_db_location)
    documents = []
    ids = []
    for index, row in file.iterrows():
        document = Document(
            page_content=row["Title"]+row["Review"],
            metadata=row["Date"]+row["Rating"],
            id=str(index)
        )
        documents.append(document)
        ids.append(str(index))
    vector_store.add_documents(documents=documents,ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k":5})