import os
from pathlib import Path

# Lê a variável de ambiente com o caminho do diretório com as configurações
configs_file_path = Path(os.getenv('APPOPTIMIZER_CONFIGS_DIR', '../configs'))


# Nome da variável que gera a depuração.
debug_variavle = os.getenv('APPOPTIMIZER_DEBUG', "False").strip().lower()

# Converte a variável para um booleano.
debug_code = debug_variavle in ("true", "1", "yes", "on", "t")
