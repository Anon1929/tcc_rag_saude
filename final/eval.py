from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_ollama.llms import OllamaLLM
import pandas as pd

from ragas import EvaluationDataset
# from ragas.metrics import FactualCorrectness
from ragas.metrics import (
    ContextPrecision,
    AnswerCorrectness
    )

import json

idTeste = 1
idPergunta = 57

respostas = {}
with open(f"experimentos/respostas/respostas_{idTeste}.json") as f:
    respostas = json.load(f)

df = pd.DataFrame([{
    "user_input": respostas['user_input'][idPergunta],
    "reference": respostas['reference'][idPergunta],
    "response": respostas['response'][idPergunta],
    "retrieved_contexts": respostas['retrieved_contexts'][idPergunta]
}])


evaluation_dataset = EvaluationDataset.from_pandas(df)


evaluator_llm = LangchainLLMWrapper(OllamaLLM(model="llama3:latest"))
embedding_model= LangchainEmbeddingsWrapper(FastEmbedEmbeddings(model_name='intfloat/multilingual-e5-large'))

result = evaluate(dataset=evaluation_dataset,metrics=[
    ContextPrecision(),
    AnswerCorrectness()
    ],llm=evaluator_llm, embeddings=embedding_model)
df = result.to_pandas()


df.to_csv("experimentos/evalNovo/resultado.csv", sep='|')