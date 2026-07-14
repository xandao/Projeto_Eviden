import os
from pathlib import Path

# Lê a variável de ambiente com o caminho do diretório com as configurações
base_files_path_env_name = 'APPOPTIMIZER_BASE_PATH'
appoptimizer_base_path = os.getenv(base_files_path_env_name, None)
base_files_path = None if appoptimizer_base_path is None else Path(appoptimizer_base_path)

# Lê a variável de ambiente com o caminho do diretório com as configurações
configs_files_dir_env_name = 'APPOPTIMIZER_CONFIGS_DIR'
configs_files_dir = Path(os.getenv(configs_files_dir_env_name, 'configs'))

# Nome da variável que gera a depuração.
sebug_variable_env_name = 'APPOPTIMIZER_DEBUG'
debug_variable = os.getenv(sebug_variable_env_name, "False").strip().lower()

# Converte a variável para um booleano.
debug_code = debug_variable in ("true", "1", "yes", "on", "t")
