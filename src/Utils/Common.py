import os
import argparse
from pathlib import Path

# Nome da variável de ambiente com o caminho do diretório base dos 
# scripts de treinamento e de otimização, usado para acessar todos os 
# subdiretórios e arquivos.
base_files_path_env_name = 'APPOPTIMIZER_BASE_PATH'

# Tenta ler a variável de ambiente base_files_path_env_name. Se a 
# variável de ambiente existir, define appoptimizer_base_path como o 
# conteúdo desta variável, que supomos ser o caminho do diretório base
# e, em cado contrário, define a variável como None.
appoptimizer_base_path = os.getenv(base_files_path_env_name, None)

# Se a variável appoptimizer_base_path for None, então base_files_path 
# também será None. Porém, se appoptimizer_base_path for uma string 
# (lida da variável de ambiente), base_files_path será um objeto Path 
# definido pelo caminho dado em appoptimizer_base_path.
base_files_path = (
    None
    if appoptimizer_base_path is None
    else Path(appoptimizer_base_path)
)

# Nome da variável de ambiente com o nome do diretório base com todas as
# configurações (usado para ler o primeiro arquivo de configuração, o
# do sistema).
configs_files_dir_env_name = 'APPOPTIMIZER_CONFIGS_DIR'

# Se a variável de ambiente APPOPTIMIZER_CONFIGS_DIR existir, permite 
# mudar o nome default do diretório de configurações de "configs" para
# o definido pela variável. A variável configs_files_dir será um objeto 
# Path para o nome do diretório definido por configs_files_dir_env_name
# ou para o diretório default "configs".
configs_files_dir = Path(os.getenv(configs_files_dir_env_name, 
                                   'configs'))

# Nome da variável de ambiente que define se depuração deve ser usada.
debug_variable_env_name = 'APPOPTIMIZER_DEBUG'

# Se a variável de ambiente APPOPTIMIZER_DEBUG exitir, retorna o valor 
# dela em debug_variable. Em caso contrário, define a variável 
# debug_variable para a string "false". Depois, converte o valor lido da
# variável ou o valor default "False", para uma string com todas as
# letras em minísculas.
debug_variable = (
    os.getenv(debug_variable_env_name, "False")
    .strip()
    .lower()
)

# Converte o valor da variável debug_variable para true ou false. A 
# variável debug_code será True se a string em debug_variable for 
# "true", "1", "yes", "on" ou "t" (e todas as variantes com mistura de 
# letras maiúsculas e minísculas devido à conversão para letras 
# minísculas feita ao definir debug_variable), e para False se for 
# alguma outra string diferente das anteriores.
debug_code = debug_variable in ("true", "1", "yes", "on", "t")
class CustomFormatter(argparse.RawTextHelpFormatter):
  """
  Classe para fazer uma formatação customizada, derivada da classe
  argparse.RawTextHelpFormatter, que converte todos os "\n" 
  para quebras de linha nas ajudas do argparse. A derivação foi feita
  para permitir mudar o texto em ingles "usage:" para "uso:".

  """
  def _format_usage(self, usage, actions, groups, prefix):
    """
      Função para formatar a linha usage. Estamos substituindo a função
      da classe pai _format_usage para alterar o seu comportamento.

      Parâmetros:
        usage: Parâmetro passado a função _format_usage da classe pai 
               argparse.RawTextHelpFormatter, que é o usage passado ao 
               construtor argparse.ArgumentParser.     
        actions: Parâmetro passado a função _format_usage da classe pai
                 argparse.RawTextHelpFormatter, que é uma lista com as 
                 opções denifidas no parser usando o add_argument.     
        groups: Parâmetro passado a função _format_usage da classe pai
                argparse.RawTextHelpFormatter, e parece que tem as 
                informações de divisão dos parâmetros e, grupos.     
        prefix: Parâmetro passado a função _format_usage da classe pai 
                argparse.RawTextHelpFormatter, com o prefixo "usage: "
                da ajuda, se for definido, que mudaremos para "uso: "
    """
    # Se nenhum prefixo for usado, o padrão do Python seria usar 
    # "usage: "". O código a seguir garante que será usado "uso:".
    if prefix is None:
        prefix = "uso: "
        
    # Chama o método original do argparse.RawTextHelpFormatter passando
    # o prefixo em português para continuar com a formatação da ajuda.
    return super()._format_usage(usage, actions, groups, prefix)