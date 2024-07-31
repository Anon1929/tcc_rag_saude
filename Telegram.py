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
import RAG
import RAGmem
import GD

import logging

from telegram import ForceReply, Update,InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters,CallbackQueryHandler

load_dotenv() 


# Abre VectorStore
vectorStore = Chroma(embedding_function=FastEmbedEmbeddings() ,persist_directory='db')
#chat = RAG.RAG(vectorStore, "llama3:latest")
chat = RAGmem.RAGmem(vectorStore, "llama3:latest")


GD.download_googledrive_folder(os.getenv("GD_DIR"),"docs",os.getenv("GD_TOKEN"),False)


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

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

    logger.info(update.message.text)
    textoResult = result["result"]
    textoResult +=  "\n\nFontes:\n"
    for source in result["source_documents"]:
        textoResult += "página " + str(source.metadata["page"]) + " do arquivo " + source.metadata["source"] +"\n"
    await update.message.reply_text(textoResult)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Olá, como posso ajudar?")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Pergunte qualquer coisa, para atualizar a base de dados digite /atualizar")


async def debug(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.message.from_user
    print(' user {} - user ID: {} '.format(user['username'], user['id']))


def main() -> None:

    application = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    
    application.add_handler(CommandHandler("debug", debug))


    #Handler do Chat 
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,respostaChat ))

    #Botão
    # application.add_handler(CallbackQueryHandler(button))

    print("Iniciado")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()