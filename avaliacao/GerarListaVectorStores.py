from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
import sys
import os
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter


import os
import sys

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

# Parametros
splitter_types = [
    ("RecursiveCharacterTextSplitter", [["\n\n", "\n"], ["\n\n", "\n", ". "]]),
    ("CharacterTextSplitter", ["\n\n", "\n", ". "])
]
chunk_overlaps = [50, 100]
similarity_metrics = ["ip", "l2", "cosine"]
output_folder = "experimentos/dbs"
os.makedirs(output_folder, exist_ok=True)

#Splitters
def split_documents(documents, splitter, separator, chunk_overlap):
    if splitter == "CharacterTextSplitter":
        text_splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=512,
            chunk_overlap=chunk_overlap
        )
    elif splitter == "RecursiveCharacterTextSplitter":
        text_splitter = RecursiveCharacterTextSplitter(
            separators=separator,
            chunk_size=512,
            chunk_overlap=chunk_overlap
        )
    else:
        raise ValueError("Invalid splitter type.")

    return text_splitter.split_documents(documents)

def generate_vector_store(documents, metric, persist_directory):
    collection_metadata = {"hnsw:space": metric}

    # Create vector store
    embed_model = FastEmbedEmbeddings(model_name='intfloat/multilingual-e5-large')
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embed_model,
        persist_directory=persist_directory,
        collection_metadata=collection_metadata
    )
    vectorstore.persist()

if __name__ == "__main__":
    folder_path = "selectedDocs"
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' not found.")

    # Carrega Aquivos
    print("Loading and splitting documents...")
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    raw_documents = []
    for file in files:
        path = os.path.join(folder_path, file)
        loader = PyPDFLoader(path)
        raw_documents += loader.load()
        print("Arquivo Carregado "+ path)
    
    db_index = 1

    # Iterar
    metadata_file = os.path.join(output_folder, "metadata.txt")
    with open(metadata_file, "w") as f:
        for splitter, separators in splitter_types:
            for separator in separators:
                for chunk_overlap in chunk_overlaps:
                    if(splitter == 'RecursiveCharacterTextSplitter'): 
                        formated_separator = separator[0].replace('\n', '\\n')+" e "+separator[1].replace('\n', '\\n')
                    else:
                        formated_separator = separator.replace('\n', '\\n')


                    print(f"Splitting  {splitter}, {formated_separator}, overlap={chunk_overlap}")
                    split_docs = split_documents(raw_documents, splitter, separator, chunk_overlap)

                    for metric in similarity_metrics:
                        persist_directory = os.path.join(output_folder, f"db_{db_index}")
                        os.makedirs(persist_directory, exist_ok=True)

                        print(f" vector store db_{db_index} com metric: {metric}")
                        generate_vector_store(split_docs, metric, persist_directory)

                        # Record
                        f.write(f"db_{db_index}: Splitter={splitter}, Separator={formated_separator}, ChunkOverlap={chunk_overlap}, Metric={metric} \n")
                        db_index += 1


    print(f"Vector stores em '{output_folder}'.")
