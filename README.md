
# Preparando o ambiente

Tenha o python e o ollama instalados. O projeto foi desenvolvido utilizando o python3.10.12

instale as dependências python do projeto com `pip install -r requirements.txt`

# Bot de Telegram

Para acionar o bot de telegram rode `python Telegram.py`
\\
O bot faz chamadas periódicas para o servidor do telegram verificando a chegada de novas mensagens
\\
Para utilizá-lo, procure o nome "Agente_ComunitarioBot" (sem as aspas) no telegram

## Comandos do bot

### /start

Cumprimenta o usuário e dá instruções básicas de uso

### /help

Mostra os comandos disponíveis

### /vectorStore

Remonta o vectorStore com os documentos que estiverem na pasta

### Mensagens comuns

Mensagens comuns serão enviadas como perguntas para o RAG e respondidas

# CLI

Para usar o CLI, rode o comando `python main.py`

Ao iniciar, irá te perguntar se deseja gerar o VectorStore novamente

Após isso, qualquer input será interpretado como pergunta para o RAG
