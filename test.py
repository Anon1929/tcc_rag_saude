from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import create_history_aware_retriever
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history")

vectorStore = Chroma(embedding_function=FastEmbedEmbeddings() ,persist_directory='db')
llm = Ollama(model="llama3:latest")


template = """"
    Histórico da conversa:
    {chat_history}
    pergunta: {input}
"""

QA_CHAIN_PROMPT = PromptTemplate(
                input_variables=["chat_history", "input"],
                template=template,
            )

retriever = vectorStore.as_retriever()
chat_retriever_chain = create_history_aware_retriever(
    llm, retriever, QA_CHAIN_PROMPT
)

while True:
    query = input("\n>: ")
    result = chat_retriever_chain.invoke({"input": query, "chat_history": memory.chat_memory})
    print(result)
    quit()
    # print("\nChat: " + result["result"])
    # memory.chat_memory.add_user_message(query)
    # memory.chat_memory.add_ai_message(result["result"])
