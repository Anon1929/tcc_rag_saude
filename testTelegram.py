from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

import sys
import os

import generateVectorStore
import RAG


import logging

from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters



# Abre VectorStore
vectorStore = Chroma(embedding_function=FastEmbedEmbeddings() ,persist_directory='db')
chat = RAG.RAG(vectorStore, "llama3:latest")


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
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


# Define a few command handlers. These usually take the two arguments update and
# context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Olá,  {user.mention_html()}! Pergunte algo, ~Inserir texto de abertura depois",
        reply_markup=ForceReply(selective=True),
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Texto de ajuda ~TODO!")


async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Echo the user message."""
    await update.message.reply_text(update.message.text)


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token("7102297178:AAFSJ17X-zgetmwHVP7pl4rAR5wjivbqw2I").build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # on non command i.e message - echo the message on Telegram
    # application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,respostaChat ))

    print("Iniciado")
    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()