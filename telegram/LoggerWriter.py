import sys
# Classe que redireciona saída do stdout pro logger
class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = ''

    def write(self, message):
        if message != '\n':  # Evitar logging de linhas vazias
            self.logger.log(self.level, message)

    def flush(self):
        pass  # Não há necessidade de flush para o logger