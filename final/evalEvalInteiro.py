from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_ollama.llms import OllamaLLM
import pandas as pd

from ragas import EvaluationDataset
# from ragas.metrics import FactualCorrectness

from contextlib import redirect_stdout

from ragas.run_config import RunConfig

# from ragas.metrics import (
#     ContextPrecision,
#     AnswerCorrectness
#     )

from ragas.metrics import (
    Faithfulness,
    )

import json


desconsiderar = [13, 14, 15, 16, 17, 18, 22,23,24,28,29,30]
for i in range(1,31):
    if i in desconsiderar:
        continue

    respostas = {}
    with open('out.txt', 'a') as f:
        f.write(f"Iniciando eval de id {i}\n")

    with open(f"./experimentos/respostas/respostas_{i}.json") as f:
        respostas = json.load(f)

    df = pd.DataFrame(respostas)
    evaluation_dataset = EvaluationDataset.from_pandas(df)

    evaluator_llm = LangchainLLMWrapper(OllamaLLM(model="llama3:latest"))
    embedding_model= LangchainEmbeddingsWrapper(FastEmbedEmbeddings(model_name='intfloat/multilingual-e5-large'))
    my_run_config = RunConfig(max_workers=8, timeout=250,max_retries=15)



    result = evaluate(dataset=evaluation_dataset,metrics=[
        Faithfulness()
        ],llm=evaluator_llm, embeddings=embedding_model,
        run_config=my_run_config)
    df = result.to_pandas()


    df.to_csv(f"experimentos/evalfaithfulness/resultadoEvalFaith{i}.csv", sep='|')

    with open('out.txt', 'a') as f:
        f.write(f"Eval de id {i} finalizado\n")
        
