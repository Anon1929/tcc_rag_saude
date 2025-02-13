from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
import pandas as pd

if __name__ == "__main__":
    similarity_metrics = ["ip", "l2", "cosine"]

    embedder = FastEmbedEmbeddings(model_name='intfloat/multilingual-e5-large')

    embeddingSizes = {}
    
    for i in range(1, 31): 
        similarity_metric = similarity_metrics[(i-1)%3]
    
        collection_metadata={
            "hnsw:space": similarity_metric
        }
        persist_directory = f"experimentos/dbs/db_{i}" 

        vectorStore = Chroma(embedding_function=embedder,
                            persist_directory=persist_directory,
                            collection_metadata = collection_metadata)
        
        data = vectorStore.get(include=["documents", "metadatas"])

        embeddingSizes[i] = pd.Series([len(doc) for doc in data["documents"]]).describe()
    
    pd.DataFrame(embeddingSizes).T.to_csv("chunkSizes.csv")
