import os
import argparse
from pathlib import Path

# Nome da variável de ambiente com o caminho do diretório com o diretório base dos scripts de
# treinamento e de optimização, usado para acessar todos os subdiretórios e arquivos.
base_files_path_env_name = 'APPOPTIMIZER_BASE_PATH'

# Tenta ler a variávelde ambiente, definindo appoptimizer_base_path em None se a 
# variável não estiver definida.
appoptimizer_base_path = os.getenv(base_files_path_env_name, None)

# Se a variável appoptimizer_base_path for None, então base_files_path também será None.
# Porém, se appoptimizer_base_path for uma string (lida da variável de ambiente), ase_files_path
# será um objeto Path para o caminho definido por appoptimizer_base_path.
base_files_path = None if appoptimizer_base_path is None else Path(appoptimizer_base_path)

# Nome da variável de ambiente com o caminho do diretório com as configurações.
configs_files_dir_env_name = 'APPOPTIMIZER_CONFIGS_DIR'

# Se a variável de ambiente APPOPTIMIZER_CONFIGS_DIR existir, permite mudar o nome default do
# diretório de configurações "configs". A variável configs_files_dir é um objeto Path para o
# diretório escolhido das configurações.
configs_files_dir = Path(os.getenv(configs_files_dir_env_name, 'configs'))

# Nome da variável de ambiente que define se depuração deve ser usada.
sebug_variable_env_name = 'APPOPTIMIZER_DEBUG'

# Se a variável de ambiente APPOPTIMIZER_DEBUG exitir, retorna o valor dala em debug_variable.
# Em caso contrário, define a variável debug_variable para a string "false". Converte o valor
# lido da variável ou o valor default "False", para uma string com todas as letras minísculas.
debug_variable = os.getenv(sebug_variable_env_name, "False").strip().lower()

# Converte o valor da variável debug_variable para true ou false. A variável debug_code será
# true se a string em debug_variable for "true", "1", "yes", "on" ou "t" (e todas as variantes
# com letras maiúsculas e minísculas, devido à conversão para letras minísculas ao definir 
# debug_variable), e para false se for alguma outra string diferente das anteriores.
debug_code = debug_variable in ("true", "1", "yes", "on", "t")
class CustomFormatter(argparse.RawTextHelpFormatter):
  """
  Classe para fazer uma formatação customizada, como a de argparse.RawTextHelpFormatter,
  em que os \n são comvertidos para quebras de linha no help do argparse, e também para
  musar a strig em ingles "usage:" para "uso:".

  """
  def _format_usage(self, usage, actions, groups, prefix):
    """
      Função para formatar a linha usage. Estamos substituindo a função
      da classe pai _format_usage para alterar o seu comportamento.

      Parâmetros:
        usage: Parâmetro passado a função _format_usage da classe pai argparse.RawTextHelpFormatter,
               que é o usage passado ao construtor argparse.ArgumentParser.     
        actions: Parâmetro passado a função _format_usage da classe pai argparse.RawTextHelpFormatter,
                 que é uma lista com as opções denifidas no parser usando o add_argument.     
        groups: Parâmetro passado a função _format_usage da classe pai argparse.RawTextHelpFormatter,
                parece que é para agrupar os argumentos em grupos.     
        prefix: Parâmetro passado a função _format_usage da classe pai argparse.RawTextHelpFormatter, 
                com o prefixo "usage: " do help, se for definido. 
    """
    # Se nenhum prefixo for usado, o padrão do Python seria usar "usage: "". O código a
    # seguir garante que será usado "uso:".
    if prefix is None:
        prefix = "uso: "
        
    # Chama o método original do argparse.RawTextHelpFormatter passando o prefixo em português
    return super()._format_usage(usage, actions, groups, prefix)