
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
import RAG
import pandas as pd
import json

# TODO Novo objeto Rag-Embedding, que é possível alterar valores de modelo, tamanho de chunk do embedding, etc por fora

vectorStore = Chroma(embedding_function=FastEmbedEmbeddings() ,persist_directory='db')
chat = RAG.RAG(vectorStore, "llama3:latest")
df = pd.read_csv("DataSetPerguntas.csv",sep = '|')
answerlist = []
contextlist = []

for index, row in df.iterrows():
    question = row["question"]

    response = chat.invoke(question)
    answerlist.append(response['result'])
    contextlist.append([document.page_content for  document in response['source_documents']])

dictResult = df[['question', 'ground_truths']].to_dict('list')
dictResult['user_input'] = dictResult.pop['question']
dictResult['reference'] = dictResult.pop['ground_truths']

dictResult['response'] = answerlist
dictResult['retrieved_contexts'] = contextlist

print(dictResult)
with open('DataSetDictPerguntasRespostas.json', 'w', encoding='utf8') as fp:
    json.dump(dictResult, fp,ensure_ascii=False,indent=4)