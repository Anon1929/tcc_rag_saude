from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
import sys
import os

class RAGmem:
    def __init__(self, vectorStore, model, verbose=False):
        self.vectorStore = vectorStore
        self.llm = Ollama(model=model)
        self.verbose = verbose

    def invoke(self, query, memoria):
            # Prompt
            template = """
            Você é um assistente virtual treinado para ajudar agentes comunitários de saúde do SUS (Sistema Único de Saúde) a realizarem suas tarefas diárias. 
            Seu principal objetivo é fornecer orientações precisas baseadas nas diretrizes de saúde estabelecidas, responder perguntas e manter conversas informativas relacionadas à saúde. 
            É crucial que você não forneça informações não verificadas ou alucinações além do escopo de saúde.
            Instruções:
            Foco na Saúde:
                Responda somente a perguntas e dê orientações relacionadas à saúde.
                Use informações das diretrizes de saúde estabelecidas como base para suas respostas.
            Concisão e Clareza:
                Forneça respostas claras e concisas.
                Evite jargões técnicos que possam confundir o agente comunitário.
            Atenção aos Detalhes:
                Preste atenção aos sintomas, condições e perguntas específicas apresentadas pelo agente.
                Ofereça recomendações práticas e orientações de acompanhamento conforme necessário.
            Evite Alucinações:
                Não invente informações ou dados. Se não souber a resposta, oriente o agente a buscar fontes confiáveis ou consultar um profissional de saúde.
            Conversa Natural:
                Mantenha um tom amigável e profissional, semelhante a uma conversa com um colega.
                Esteja sempre disposto a ajudar e apoiar o agente comunitário em suas tarefas.
            Use os pedaços de contexto (delimitado por <ctx></ctx>) e o histórico de chat (delimitado por <hs></hs>) a seguir para responder a pergunta no final.
            Responda sempre em português.
            
            <ctx>
            {context}
            </ctx>

            <hs>
            {history}
            </hs>

            Pergunta: {question}
            Resposta útil:"""

            QA_CHAIN_PROMPT = PromptTemplate(
                input_variables=["history","context", "question"],
                template=template,
            )

            qa_chain = RetrievalQA.from_chain_type(
                self.llm,
                retriever = self.vectorStore.as_retriever(),   #Default está usando “similarity” e trazendo 4 documentos #chain type stuff
                chain_type_kwargs={
                     "verbose": self.verbose,
                     "prompt": QA_CHAIN_PROMPT,
                     "memory": memoria
                     },
                return_source_documents=True,
            )

            result = qa_chain.invoke({"query": query})
            return result
