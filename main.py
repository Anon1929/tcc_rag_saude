from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
import sys
import os

import generateVectorStore
import RAG

x = input("Deseja gerar o VectorStore?(S/N)")
if (x=='s' or x=='S'):
    vectorStore = generateVectorStore.generate_vector_store("docs")
else:
    vectorStore = Chroma(embedding_function=FastEmbedEmbeddings() ,persist_directory='db')


chat = RAG.RAG(vectorStore, "llama3:latest")

while True:
    query = input("\nEu: ")
    if query == "exit":
        break
    if query.strip() == "":
        continue

    result = chat.invoke(query)
    print("\nChat: " + result["result"])
    print("\nFontes:")
    for source in result["source_documents"]:
        print("página " + str(source.metadata["page"]) + " do arquivo " + source.metadata["source"])
