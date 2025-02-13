import json
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Ollama
from langchain_community.embeddings import FastEmbedEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_similarity,
    answer_correctness
)
from datetime import datetime
import pandas as pd
import traceback

def log(text):
    timeStr = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    finalString = timeStr + " - " + text + "\n"
    
    with open("experimentos/evals/logs.txt", "a") as f:
        f.write(finalString)
    print(finalString, end="")

def avaliar(inputFile, outputFile, llm, embedding_model):
    f = open(inputFile) 
    dataset = json.load(f) 
    f.close()
    
    ragas_testset = Dataset.from_dict(dataset)
    
    log("iniciando avaliacao")

    result = evaluate(
        dataset = ragas_testset, 
        metrics=[
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
            answer_similarity,
            answer_correctness
        ],
        llm = llm,
        embeddings= embedding_model
    )
    log("avaliacao finalizada")
    
    df = result.to_pandas()

    df.to_csv(outputFile, sep='|')
    log(f"resultado salvo no arquivo: {outputFile}")
    
if __name__ == "__main__":
    log("Iniciando execucao")
    evaluator_llm = LangchainLLMWrapper(Ollama(model="llama3:latest"))
    embedding_model= LangchainEmbeddingsWrapper(FastEmbedEmbeddings(model_name='intfloat/multilingual-e5-large'))

    try:
        for i in range(14, 31):
            inputFile = f"experimentos/respostas/respostas_{i}.json"
            outputFile = f"experimentos/evals/eval_{i}.csv"
            
            log(f"experimento {i}")
            log(f"inputFile: {inputFile}")
            log(f"outputFile: {outputFile}")

            avaliar(inputFile, outputFile, evaluator_llm, embedding_model)
    except Exception as e:
        error_details = traceback.format_exc()
        log(f"[ERROR] {error_details}")
        exit()
            
    

    