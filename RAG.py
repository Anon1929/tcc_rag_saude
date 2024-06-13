from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
import sys
import os

class RAG:
    def __init__(self, vectorStore, model):
        self.vectorStore = vectorStore
        self.llm = Ollama(model=model)

    def invoke(self, query):
            # Prompt
            template = """Use os pedaços de contexto a seguir para responder a pergunta no final.
            Se você não souber a resposta, só diga que não sabe, não tente inventar uma resposta.
            Use no máximo 3 frases e mantenha as respostas o mais concisas possível. Responda sempre em português.
            {context}
            Pergunta: {question}
            Resposta útil:"""

            QA_CHAIN_PROMPT = PromptTemplate(
                input_variables=["context", "question"],
                template=template,
            )

            qa_chain = RetrievalQA.from_chain_type(
                self.llm,
                retriever = self.vectorStore.as_retriever(),
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
                return_source_documents=True,
            )

            result = qa_chain.invoke({"query": query})
            return result
