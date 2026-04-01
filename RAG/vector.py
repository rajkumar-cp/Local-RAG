import os
from pathlib import Path

import pandas
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, CSVLoader, JSONLoader, PyPDFLoader, Docx2txtLoader, \
    UnstructuredExcelLoader, UnstructuredWordDocumentLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

LOADER_MAPPING = {
    '.txt': TextLoader,
    '.csv': CSVLoader,
    '.json': lambda file_path: JSONLoader(file_path, jq_schema='.', text_content=False),
    '.pdf': PyPDFLoader,
    '.docx': Docx2txtLoader,
    '.xls': UnstructuredExcelLoader,
    '.xlsx': UnstructuredExcelLoader,
    '.doc': UnstructuredWordDocumentLoader
}

def load_docs_from_directory(directory_path):
    documents = []
    for file_path in Path(directory_path).rglob('*'):
        if file_path.suffix in LOADER_MAPPING:
            loader_class = LOADER_MAPPING[file_path.suffix]
            loader_instance = loader_class(str(file_path)) if not callable(loader_class) else loader_class(str(file_path))
            documents.extend(loader_instance.load())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents=documents)
    return split_docs

#
# file = pandas.read_csv("References/realistic_restaurant_reviews.csv")
embeddingModel = OllamaEmbeddings(model="mxbai-embed-large")
vector_db_location = "./vector-db"
# documents = []
# ids = []
# add_docs = not os.path.exists(vector_db_location)
# if add_docs:
#     os.makedirs(vector_db_location)
#     for index, row in file.iterrows():
#         document = Document(
#             page_content=row["Title"]+row["Review"],
#             metadata={"rating": row["Rating"], "date": row["Date"]},
#             id=str(index)
#         )
#         documents.append(document)
#         ids.append(str(index))
vector_store = Chroma(
    collection_name="Project_ABC",
    embedding_function=embeddingModel,
    persist_directory=vector_db_location
)

documents = load_docs_from_directory("./RAG/Resources")
ids = [f"doc_{i}" for i in range(len(documents))]
vector_store.add_documents(documents=documents, ids=ids)

#if add_docs:
vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k":5})