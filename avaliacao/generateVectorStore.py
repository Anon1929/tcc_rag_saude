from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

import sys
import os

from langchain.text_splitter import CharacterTextSplitter

class SuppressStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr


def generate_vector_store(folder_path: str):
    # Get a list of all files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=500, 
        chunk_overlap=50)

    documents = []

    for file in files:
        path = folder_path + "/" + file
        loader = PyPDFLoader(path)
        documents += loader.load_and_split(text_splitter)
        print("Arquivo "+path+" Carregado")
    
    collection_metadata={
        "hnsw:space": "ip"
    }

    # create the vector store
    embed_model = FastEmbedEmbeddings(model_name='intfloat/multilingual-e5-large')

    print("Gerando Vector Store ...")
    vectorstore = Chroma.from_documents(documents=documents,
        embedding=embed_model,
        persist_directory='db2',
        collection_metadata = collection_metadata
        )
    return vectorstore

if __name__ == "__main__":
    # print("Enter the path to the folder containing the PDF files:")
    # folder_path = input()
    folder_path = "selectedDocs"
    generate_vector_store(folder_path)
    print("Vector store gerado.")
