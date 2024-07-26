from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

import sys
import os
from dotenv import load_dotenv, dotenv_values 

import generateVectorStore
import RAG
import GD

import logging

from telegram import ForceReply, Update,InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters,CallbackQueryHandler

load_dotenv() 


# Abre VectorStore
vectorStore = Chroma(embedding_function=FastEmbedEmbeddings() ,persist_directory='db')
chat = RAG.RAG(vectorStore, "llama3:latest")


GD.download_googledrive_folder(os.getenv("GD_DIR"),"docs",os.getenv("GD_TOKEN"),False)


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


async def respostaChat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    result = chat.invoke(update.message.text)
    logger.info(update.message.text)
    textoResult = result["result"]
    textoResult +=  "\n\nFontes:\n"
    for source in result["source_documents"]:
        textoResult += "página " + str(source.metadata["page"]) + " do arquivo " + source.metadata["source"] +"\n"
    await update.message.reply_text(textoResult)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # keyboard = [
    #     [
    #         InlineKeyboardButton("Agente", callback_data="Agente"),
    #         InlineKeyboardButton("Mantenedor", callback_data="Mantenedor"),
    #     ]
    # ]
    # reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Olá, como posso ajudar?")

# async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

#     """Parses the CallbackQuery and updates the message text."""
#     query = update.callback_query
#     await query.answer()

#     await query.edit_message_text(text=f"Opção Selecionada: {query.data}")
#     if(query.data=="Agente"):
#         await query.message.reply_text("Permissões de anexo concedidas, caso deseje, anexe um novo documento para a base de dados.")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Pergunte qualquer coisa, para atualizar a base de dados digite /atualizar")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    await update.message.reply_text(update.message.text)


def main() -> None:

    application = Application.builder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    application.add_handler(CommandHandler("atualizar", help_command))
    application.add_handler(CommandHandler("reprocessar", help_command))
    

    #Handler do Chat
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,respostaChat ))

    #Botão
    # application.add_handler(CallbackQueryHandler(button))

    print("Iniciado")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()