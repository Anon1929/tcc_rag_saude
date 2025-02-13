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
    context_entity_recall,
    answer_similarity,
    answer_correctness
)

f = open('DataSetDictPerguntasRespostas.json') 
dataset = json.load(f) 
f.close()


ragas_testset = Dataset.from_dict(dataset)

evaluator_llm = LangchainLLMWrapper(Ollama(model="llama3:latest"))
embedding_model= LangchainEmbeddingsWrapper(FastEmbedEmbeddings(model_name='intfloat/multilingual-e5-large'))

result = evaluate(
    dataset = ragas_testset, 
    metrics=[
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
        context_entity_recall,
        answer_similarity,
        answer_correctness
    ],
    llm = evaluator_llm,
    embeddings= embedding_model
)

df = result.to_pandas()
print(df.head())
df.to_csv('resultRagasEval.csv',sep='|')