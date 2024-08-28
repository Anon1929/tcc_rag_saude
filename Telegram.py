from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.memory import ConversationBufferMemory

import sys
import os
from dotenv import load_dotenv, dotenv_values 

import generateVectorStore

import RAGmem
import GD

import logging
import LoggerWriter

from telegram import ForceReply, Update,InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters,CallbackQueryHandler

load_dotenv() 

vectorStore = {}
# Abre VectorStore
S = input("Deseja Gerar o vectorStore?\n")
if(S == 'S' or S =='s'):
    GD.download_googledrive_folder(os.getenv("GD_DIR"),"docs",os.getenv("GD_TOKEN"),False)
    print("Gerando Vector Store")
    vectorStore = generateVectorStore.generate_vector_store("docs")
    print("VectorStore Gerado")
else:
    vectorStore = Chroma(embedding_function=FastEmbedEmbeddings() ,persist_directory='db')


#chat = RAG.RAG(vectorStore, "llama3:latest")
chat = RAGmem.RAGmem(vectorStore, "llama3:latest")

logging.basicConfig(
    filename='logs/langchain_verbose.log',  # Nome do arquivo de log
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Formato do log)
)

logger = logging.getLogger('langchain')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)
sys.stdout = LoggerWriter.LoggerWriter(logger, logging.DEBUG)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

globalDictMemoria = {}

def memoryCheck(id):
    if(id in globalDictMemoria):
        print("Usuario com historico")
        return globalDictMemoria[id]
    else :
        print("Usuario novo")
        globalDictMemoria[id] = ConversationBufferMemory(memory_key="history",input_key="question")
        return globalDictMemoria[id]


async def respostaChat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    memUser = memoryCheck(user['id'])

    #result = chat.invoke(update.message.text)
    result = chat.invoke(update.message.text, memUser)

    logger.info("P -" + str(user['id']) + " - " + update.message.text)
    textoResult = result["result"]
    textoResult +=  "\n\nFontes:\n"

    if(len(result["source_documents"]) == 0 ):
        textoResult = "Não foram encontradas fontes para a informação analisada.\nConsulte um profissional da saúde ou busque em fontes confiáveis.\nCaso possível, entre em contato com a equipe de manutenção para adicionar novas fontes."

    for source in result["source_documents"]:
        textoResult += "página " + str(source.metadata["page"]) + " do arquivo " + source.metadata["source"] +"\n"
    logger.info( "R - " + str(user['id']) + " - " + textoResult)
    await update.message.reply_text(textoResult)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Olá, eu sou um assistente virtual treinado para ajudar agentes comunitários de saúde do SUS (Sistema Único de Saúde) a realizarem suas tarefas diárias.\nMeu principal objetivo é fornecer orientações precisas baseadas nas diretrizes de saúde estabelecidas, responder perguntas e manter conversas informativas relacionadas à saúde.\n Pode fazer sua pergunta, em caso de dúvidas digite o comando /help")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Pergunte qualquer coisa, para atualizar a base de dados digite /vectorStore")


async def debug(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    print(' user {} - user ID: {} '.format(user['username'], user['id']))


async def gerarVectorStore(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Atualizando a base de dados")
    GD.download_googledrive_folder(os.getenv("GD_DIR"),"docs",os.getenv("GD_TOKEN"),False)

    vectorStore = generateVectorStore.generate_vector_store("docs")
    
    global chat
    chat = RAGmem.RAGmem(vectorStore, "llama3:latest")

    await update.message.reply_text("Base de dados atualizada")

def main() -> None:

    application = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    application.add_handler(CommandHandler("start", start))

    application.add_handler(CommandHandler("help", help_command))
    
    application.add_handler(CommandHandler("debug", debug))

    application.add_handler(CommandHandler("vectorStore", gerarVectorStore))

    #Handler do Chat 
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,respostaChat ))

    #Botão
    # application.add_handler(CallbackQueryHandler(button))

    print("Iniciado")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()