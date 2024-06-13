from langchain_community.document_loaders import PyPDFLoader

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import sys
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

    documents = []

    for file in files:
        path = folder_path + "/" + file
        loader = PyPDFLoader(path)
        documents += loader.load_and_split(text_splitter)
        print("Arquivo "+path+" Carregado")

    # create the vector store
    print("Gerando Vector Store ...")
    with SuppressStdout():
        vectorstore = Chroma.from_documents(documents=documents, embedding=FastEmbedEmbeddings(),persist_directory='db')

    return vectorstore


if __name__ == "__main__":
    # print("Enter the path to the folder containing the PDF files:")
    # folder_path = input()
    folder_path = "docs"
    generate_vector_store(folder_path)
    print("Vector store generated successfully.")
