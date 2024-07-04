from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys
import os

folder_path = "docs"
files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

documents = []
doccarregadosteste = []
for file in files:
    path = folder_path + "/" + file
    loader = PyPDFLoader(path)
    # print(loader.source)
    documento = loader.load_and_split(text_splitter)
    var =  loader.load()
    if not var:
        print("Problema em carregar")
    if len(var)==0:
        print("Problema 2 em carregar")
    # doccarregadosteste += loader.load()
    print("Arquivo "+path+" Carregado "+ str(len(var))+ " - " + str(len(documento)))
# print(len(documents))
# print(len(doccarregadosteste))