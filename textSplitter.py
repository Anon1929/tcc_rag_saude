from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import CharacterTextSplitter

import pandas as pd
import os
import random
def splitText(path: str, text_splitter):
    documents = []

    loader = PyPDFLoader(path)
    documents += loader.load_and_split(text_splitter)

    return documents

if __name__ == "__main__":
    folder_path = "selectedDocs"
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    text_splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=500, 
        chunk_overlap=50)

    contextos = []
    fontes = []
    pages = []

    for file in files:
        path = "./selectedDocs/" + file
        documents = splitText(path, text_splitter)
        random_indices = random.sample(range(len(documents)), k=len(documents)//20)    
        print("Arquivo "+path+" Carregado")
        for i in random_indices:
            contextos.append(documents[i].page_content)
            fontes.append(file)
            pages.append(0)
    df = pd.DataFrame({
        "filename": fontes,
        "page": pages,
        "text": contextos
    })

    df.to_csv("ContextosSelecionados.csv",sep = '|', index=True)
