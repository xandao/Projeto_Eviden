import os
from pathlib import Path

# Lê a variável de ambiente com o caminho do diretório com as configurações
APPOPTIMIZER_BASE_PATH = os.getenv('APPOPTIMIZER_BASE_PATH', None)
base_files_path = None if APPOPTIMIZER_BASE_PATH is None else Path(APPOPTIMIZER_BASE_PATH)

# Lê a variável de ambiente com o caminho do diretório com as configurações
configs_files_dir = Path(os.getenv('APPOPTIMIZER_CONFIGS_DIR', 'configs'))

# Nome da variável que gera a depuração.
debug_variable = os.getenv('APPOPTIMIZER_DEBUG', "False").strip().lower()

# Converte a variável para um booleano.
debug_code = debug_variable in ("true", "1", "yes", "on", "t")
