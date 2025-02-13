
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
import RAG
import pandas as pd
import json
from datetime import datetime

from tqdm import tqdm

def log(text):
    timeStr = datetime.now().strftime("%H:%M:%S")
    finalString = timeStr + " - " + text + "\n"
    
    with open("experimentos/respostas/logs.txt", "a") as f:
        f.write(finalString)
    print(finalString, end="")


def gerarResposta(questions, chat, outputFile):
    answerlist = []
    contextlist = []
    log("iniciando geração de respostas")
    progress_bar = tqdm(total=len(questions)) 
    
    for index, row in questions.iterrows():
        question = row["question"]
        
        response = chat.invoke(question)
        answerlist.append(response['result'])
        contextlist.append([document.page_content for document in response['source_documents']])
        
        progress_bar.update(1) 
    
    progress_bar.close() 
    log("respostas geradas")

    dictResult = questions[['question', 'ground_truths']].to_dict('list')
    dictResult['user_input'] = dictResult.pop('question')
    dictResult['reference'] = dictResult.pop('ground_truths')

    dictResult['response'] = answerlist
    dictResult['retrieved_contexts'] = contextlist

    log(f"{len(dictResult)} dicionarios gerados")

    with open(outputFile, 'w', encoding='utf8') as fp:
        json.dump(dictResult, fp,ensure_ascii=False,indent=4)

    log(f"Resultado escrito no arquivo {outputFile}")


if __name__ == "__main__":
    similarity_metrics = ["ip", "l2", "cosine"]
    questions = pd.read_csv("questions.csv", sep = '|')

    log("iniciando execucao")

    for i in range(1, 31):
        similarity_metric = similarity_metrics[(i-1)%3]
        outputFile = f"experimentos/respostas/respostas_{i}.json"
        persist_directory = f"experimentos/dbs/db_{i}" 
        
        log(f"experimento n{i}")
        log(f"similarity_metric: {similarity_metric}")
        log(f"outputFile: {outputFile}")
        log(f"persist_directory: {persist_directory}")
    
        collection_metadata={
            "hnsw:space": similarity_metric
        }
        persist_directory = f"experimentos/dbs/db_{i}" 
        embedder = FastEmbedEmbeddings(model_name='intfloat/multilingual-e5-large')

        vectorStore = Chroma(embedding_function=embedder,
                            persist_directory=persist_directory,
                            collection_metadata = collection_metadata)
        chat = RAG.RAG(vectorStore, "llama3:latest")

        gerarResposta(questions, chat, outputFile)
