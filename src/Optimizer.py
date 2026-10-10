import pandas as pd
import numpy as np
from Utils.Suggestions import SuggestionsPredictor
from Utils.ReadConfigs import (
    ReadApplicationsConfigs,
    PredictorsInfoConfig,
    ReadSystemConfig,
    ReadUserConfig,
)
import sys
import argparse
from pathlib import Path
import os
from functools import partial
import subprocess
from Utils.Common import (
  base_files_path_env_name, 
  base_files_path, 
  configs_files_dir, 
  debug_code, 
  CustomFormatter
)
import textwrap
import tempfile
import re

def read_configs(verbose=False):
  """
  Lê os arquivos de configuração do sistema, das aplicações e do
  usuário. Retorna uma tupla com os seguintes elementos, referentes às
  configurações lidas:

  - configs_file_path: Caminho para o diretório de configuração.
  - system_config: Dicionário com as configurações do sistema.
  - applications_configs: Dicionário com as configurações das 
    aplicações.
  - user_config: Dicionário com as configurações do usuário.
  - predictors_info_config: Dicionário com as informações dos 
    preditores.

  Se algum dos arquivos de configuração não puder ser lido, a função
  retorna None para todos os elementos da tupla.

  Parâmetros:
    verbose (bool): Se True, imprime mensagens detalhadas durante a 
                    leitura dos arquivos de configuração.

  Retorna:
    tuple: Uma tupla com os elementos configs_file_path, system_config, 
           applications_configs, user_config, predictors_info_config
           desfritos anteriormente.
  """

  # Primeiramente verifica se a variável de ambiente com o caminho da 
  # base do diretório com os arquivos dos scripts está definida,
  # o que ocorrerá somente se a variável de ambiente 
  # APPOPTIMIZER_BASE_FILES_PATH estiver definida ao executar o script
  # de otimização. 
  if base_files_path is None:
    # Se a varável de ambiente não estiver definida, imprime uma
    # mensagem de erro com o nome da variável de ambiente.
    print(f"❌ Variável de ambiente {base_files_path_env_name} com o caminho "
          "da base dos scripts não foi definida")

    # Como não temos como ler os arquivos de configuração pois não
    # sabemos o diretório com as configurações, retorna None para todos
    # os elementos da tupla.
    return None, None, None, None, None
  else:
    # Como o diretório base dos arquivos dos scripts está definido,
    # descobrimos o caminho do diretório com os arquivos de
    # configuração, que será composto por este diretório base e o
    # subdiretório definido na variável configs_files_dir (este nome de
    # subdiretório é definido pela variável de ambiente 
    # APPOPTIMIZER_CONFIGS_DIR, que pode ser definida pelo adminstrador
    # do sistema para mudar o valor default "configs" usado quando a
    # variável não for definida.
    configs_file_path = base_files_path / configs_files_dir

  # Lê as configurações do sistema, que são usadas para ler as
  # configurações das aplicações, pois é neste arquivo que está o nome
  # do diretório com as configurações das aplicações. O caminho do
  # arquivo de configuração do sistema é composto pelo diretório de 
  # configuração e o nome do arquivo de configuração do sistema, que é 
  # definido como "system_config.json". O nome está atualmente fixo no
  # código, mas podemos no futuro permitir que seja definido por uma
  # variável de ambiente.
  system_config_file_path = configs_file_path / 'system_config.json'

  # Lê as configurações do sistema usando a classe ReadSystemConfig, 
  # usando a função read_system_config, para ler o arquivo de
  # configuração do sistema e retorna um dicionário com as configurações
  # lidas. Se o arquivo não puder ser lido, a função retorna None.
  system_config = ReadSystemConfig(verbose).read_system_config(
    system_config_file_path)

  # Verifica se as configurações do sistema foram lidas com sucesso, 
  # pois são necessárias para ler os arquivos de configuração das
  # aplicações, já que o nome do diretório com os arquivos de
  # configuração das aplicações está definido no arquivo de configuração
  # do sistema.
  if system_config is None:
    # Se não foi possível ler as configurações do sistema, não podemos
    # ler as configurações das aplicações, porque não sabemos o nome do
    # diretório com as configurações das aplicaçõesm, dentro do
    # diretório de configuração. Neste caso, definimos 
    # applications_configs como None, para indicar que não foi possível
    # ler as configurações das aplicações.
    applications_configs = None
  else:    
    # Para ler as configurações das aplicações, precisamos do caminho do
    # diretório com os arquivos de configuração das aplicações, que é
    # composto pelo diretório de configuração e o subdiretório definido
    # na chave "applications_path" do dicionário system_config definido
    # pelo arquivo de configuração do sistema. 
    applications_configs_dir_path = (
        configs_file_path / system_config["applications_path"]
    )

    # Lê as configurações das aplicações usando a classe
    # ReadApplicationsConfigs, para processar, usando a função 
    # read_applications_config desta classe, o diretório de configuração
    # das aplicações e retorna um dicionário com as configurações lidas, 
    # caso o diretório possa ler lido, tenha os arquivos de configuração 
    # das aplicações e estes possam ser lidos sem erros. Se algum erro
    # ocorrer durante o processamento do diretório ou dos arquivos de
    # configuração das aplicações, a função retorna None.
    applications_configs = ReadApplicationsConfigs(
      verbose
      ).read_applications_config(applications_configs_dir_path)

  # Define o caminho do arquivo de configuração do script de otimizaçao,
  # usado pelos usuários, que é composto pelo diretório de configuração
  # e o nome do arquivo de configuração do script de otimização, que é
  # definido como "user_config.json". O nome está atualmente fixo no 
  # código, mas podemos no futuro permitir que seja definido por uma
  # variável de ambiente ou no arquivo de configuração do sistema.
  user_config_file_path = configs_file_path / 'user_config.json'

  # Lê as configurações do script do usuário, usando a classe
  # ReadUserConfig, que lê o arquivo de configuração do usuário
  # como o nome do script de submissão default gerado se o usuário não
  # especificar um nome para o script e outras configurações do script,
  # como o nome do programa de submissão do script (default para sbatch)
  # e a expressão regular para extrair o ID do trabalho submetido da
  # saída do programa de submissão. 
  # TODO: Existem algumas configurações do script do usupario que ainda
  # não foram usadas, como as dadas na chave 'collect_consumed_energy'
  # para coletar o consumo de energia do job submetido, 
  # que ainda não foi implementado (seria similar ao que fizemos ao
  # obter o consumo de energia nas execuções do RAxML e dos benchmarks
  # do NAS) e a chave 'users_activity', um dicionario com as informações
  # para coletar os dados da melhor sugestão de configuração e os
  # parâmetros da aplicação, o tempo de execução estimado e as 
  # informações sobre o trabalho, ou seja, o seu nome e o JobID, e 
  # salva-los em um arquivo CSV, sendo que um arquivo será gerado para
  # cada aplicação das que podem ser otimizadas.
  user_config = ReadUserConfig(verbose).read_user_config(
    user_config_file_path)

  # Verifica se as configurações do sistema foram lidas com sucesso, 
  # pois o nome do diretório com os preditores e o nome do arquivo de
  # configuração dos preditores estão definidos no arquivo de
  # configuração do sistema.
  if system_config is None:
    # Se não foi possível ler as configurações do sistema, não podemos
    # ler o arquivo com os nomes dos arquivos dos preditores, pois não
    # temos o diretório dos preditores nem o nome do arquivo de 
    # configuração dos preditores, portanto definimos
    # predictors_info_config como None, para indicar que não foi
    # possível ler as informações dos preditores.
    predictors_info_config = None
  else:
    # Define o caminho do arquivo de configuração com as informações dos
    # preditores, que é composto pelo diretório de configuração, o
    # subdiretório definido na chave "predictors_path" do dicionário
    # system_config e o nome do arquivo de configuração dos preditores
    # definido na chave "predictors_info_config_filename" do dicionário
    # system_config.
    predictors_info_file_path = (
        base_files_path
        / Path(system_config["predictors_path"])
        / system_config["predictors_info_config_filename"]
    )

    # Lẽ o dicionário com as informações dos preditores, que é usado 
    # para ler os preditores usados para fazer a sugestão da melhor
    # configuração de execução das aplicações. Cada aplicação tem um
    # preditor, com o nome dele associado a uma chave do dicionário,
    # igual ao nome da aplicação e o valor sendo o nome do arquivo que 
    # armazena o preditor no diretório de preditores. O caminho do 
    # arquivo de configuração com as informações dos preditores é
    # composto pelo diretório de configuração, o subdiretório definido
    # na chave "predictors_path" do dicionário system_config e o nome do
    # arquivo de configuração dos preditores definido na chave 
    # "predictors_info_config_filename" do dicionário system_config. 
    predictors_info_config = PredictorsInfoConfig(
      verbose
    ).read_predictors_info_config(
      predictors_info_file_path
    )

  # Retorna uma tupla com cinco elementos, sendo o primeiro o caminho do
  # diretório de configuração, o segundo o dicionário com as
  # configurações do sistema, o terceiro o dicionário com as 
  # configurações das aplicações, o quarto o dicionário com as
  # configurações do script de otimização e o quinto o dicionário com as
  # informações dos preditores. Se algum dos arquivos de configuração 
  # não pode ser lido, a função retorna None para o arquivo que não foi
  # lido.
  return (configs_file_path, system_config, applications_configs, user_config,
          predictors_info_config)

def process_script_args():
  """
  Função para processar a linha de comando do script de otimização. A 
  linha de comando é dividida em duas partes, de acordo com o separador
  '--', a parte antes do separador, os parâmetros do script de
  otimização, e os parâmetros após o separador, o nome da aplicação 
  seguido pelos parâmetros da aplicação. Os parâmetros do scipt, da
  aplicação e o nome são todos separados por espaços. No momento, temos
  somente os seguintes parâmetros:
  # TODO: Eu estou pensando, no caso dos parâmetros relacionados as
  # variáveis de configuração, mostrar somente as variáveis usadas pela
  # aplicação que está sendo otimizada. Isso necessitaria mudar o código
  # para detectar a aplicação nesta função. Eu acho adequado, porque
  # além de poder fazer isso, poderíamos concentrar todo o código de
  # processamento dos parâmetros, incluindo o da aplicação, nesta
  # função, além de já fazer o tratamento dos parâmetros -n, -p e -t e
  # os outros que serão similares.

  -r, --run: Submete o script com a melhor configuração de execução.
  -j, --jobname: Nome do trabalho registrado no sistema de submissão.
  -s, --script: Salva o script gerado em um arquivo.
  -S, --suggestion: Mostra somente a sugestão para os parâmetros do
                    script.
  -n, --nodes: Lista com os possíveis números de nós, se a aplicação usa
               múltiplos nós. Usada conjuntamente com as opções -p e -t.
  -p, --process: Lista com os possíveis números de processos, se a 
                 aplicação usa múltiplos processos por nó. Usada
                 conjuntamente com as opções -n e -t. 
  -t, --threads: Lista com os possíveis números de threads, se a 
                 aplicação usa múltiplas threads por processo. Usada 
                 conjuntamente com as opções -n e -p.
  -l, --list: Lista as aplicações cujas execuções podem ser otimizadas 
              pelo script.
  -v, --verbose: Habilita a verbosidade com informações sobre a melhor
                 sugestão de configuração escolhida para a aplicação.

		Parâmetros:
      Não tem parâmetros.

    Retorna:
      tupla: Uma tupla com dois elementos, o primeiro é as informações
             sobre os parâmetros do script de otimização descritos 
             anteriormente em uma tupla nomeada chamada de NameSpace,
             e o segundo parâmetro retorna os parâmetros da aplicação em
             uma lista, sendo a primeira entrada o nome da aplicação.
  """
  # Inicializa o analisador dos parâmetros do script de otimização, com
  # uma descrição, uma descrição do formato da linha de comando do
  # script. Também informamos que vamos definir uma ajuda (help)
  # customizada, necessário para escrever a ajuda em português, e uma
  # classe de formatação, necessário para lidar com múltiplas linhas e,
  # no nosso caso, que definimos um objeto da classe customizada 
  # CustomFormatter, mudar o nome na drescição da linha de comando do
  # inglês "usage:" para o português "uso".
  parser = argparse.ArgumentParser(
    description="Script para escolher a melhor configuração para as "
                "aplicações selecionadas.",
    usage="script_optimizer [opções] -- [executável da aplicação] [-h] ["
          "opções obrigatórias da aplicação] [outras opções da aplicação]",
    add_help=False,
    formatter_class=CustomFormatter
  )

  # Cria um objeto cuja referência será armazenada em opcoes para 
  # agrupar as opções gerais do script, sendo que a string 
  # "Opções principais" passada ao criar o objeto será o texto mostrado
  # antes dos textos com as descrições de cada opção do script, com 
  # exceção da ajuda, que será colocada em um outro grupo único.
  opcoes = parser.add_argument_group("Opções principais")

  # Cria um objeto cuja referência será armazenada em ajuda para criar 
  # um grupo somente com a opção de ajuda, sendo que a string  "Ajuda" 
  # passada ao criar o objeto será o texto mostrado antes do texto da 
  # opção de ajuda.
  ajuda = parser.add_argument_group("Ajuda")

  # Adiciona ao grupo de opções gerais, referenciado por opcoes, a opção
  # -r ou --run que permite ao usuário executar o script gerado com a 
  # melhor sugestão de configuração para a aplicação escolhida pelo
  # usuário.
  opcoes.add_argument(
    "-r", "--run", 
    action="store_true", 
    default=False, 
    help="Submete o script com a melhor configuração de execução."
  )

  # Adiciona ao grupo de opções gerais, referenciado por opcoes, a opção
  # -j ou --jobname que permite ao usuário dar um nome para o trabalho a
  # ser submetido pelo script gerado com a melhor sugestão de
  # configuração para a aplicação escolhida pelo usuário.
  opcoes.add_argument(
    "-j", "--jobname", 
    type=str, 
    default=None, 
    help="Nome do trabalho registrado no sistema de submissão."
  )
  
  # Adiciona ao grupo de opções gerais, referenciado por opcoes, a opção
  # -s ou --script que permite ao usuário definir o caminho e o nome do
  # arquivo em que será salvo o script gerado a melhor sugestão de
  # configuração para a aplicação escolhida pelo usuário.
  opcoes.add_argument(
    "-s", "--script", 
    type=str, 
    default=None, 
    help="Salva o script gerado em um arquivo."
  )

  # Adiciona ao grupo de opções gerais, referenciado por opcoes, a opção
  # -S ou --suggestion que permite ao usuário desabilitar a geração do
  # script de submissão, sendo que a melhor sugestão de configuração
  # será mostrada no terminal em que o usuário executou o script.
  opcoes.add_argument(
    "-S", "--suggestion", 
    action="store_true", 
    default=False, 
    help="Mostra somente a sugestão para os parâmetros do script."
  )

  # Adiciona ao grupo de opções gerais, referenciado por opcoes, a opção
  # -n ou --nodes que permite ao usuário definir os possíveis valores da
  # variável de configuração que define o números de nós. O uso da opção
  # habilita o uso das configurações otimizadas, e as outras opções
  # referentes às outras variáveis de configuração serão definidas em
  # '1' se não forem definidas pelo usuário.
  opcoes.add_argument(
    "-n", "--nodes", 
    type=str, 
    nargs="+", 
    default=None, 
    help=textwrap.dedent
    (
      """\
        Lista com os possíveis números de nós, se a aplicação usa múltiplos "
        "nós.
        Usada conjuntamente com as opções -p e -t, que terão os valores "
        "padrão se não forem usadas.
        Cada elemento da lista está no formato i:e:s, onde i é o número "
        "inicial, f é o final e s é o passo.
        Pode-se omitir o i, que será igual a 1, o e, que será igual a i, e o "
        "s, que será igual a 1.
        Default 1:1.

          Exemplos: -n 1 2:10:2 -> Nós: 1, 2, 4, 6, 8, 10.
                    -n :10:2    -> Nós: 1, 3, 5, 7, 9.
                    -n 1:5      -> Nós: 1, 2, 3, 4, 5.                                                                 
      """
     )
  )                      

  # Adiciona ao grupo de opções gerais, referenciado por opcoes, a opção
  # -p ou --process que permite ao usuário definir os possíveis valores
  # da variável de configuração que define o números de processos por
  # nó. O uso da opção habilita o uso das configurações otimizadas, e as
  # outras opções referentes às outras variáveis de configuração serão
  # definidas em '1' se não forem definidas pelo usuário.
  opcoes.add_argument(
    "-p", "--process", 
    type=str, 
    nargs="+", 
    default=None, 
    help=textwrap.dedent(
      """\
        Lista com os possíveis números de processos, se a aplicação usa "
        "múltiplos processos por nó."
        Usada conjuntamente com as opções -n e -t, que terão os valores "
        "padrão se não forem usadas."
        Cada elemento da lista está no formato i:e:s, onde i é o número "
        "inicial, f é o final e s é o passo."
        Pode-se omitir o i, que será igual a 1, o e, que será igual a i, e o "
        "s, que será igual a 1."
        Default 1:1.

        Exemplos: -p 1 2:      -> Processos: 1, 2.
                  -p 1 :3:1    -> Processos: 1, 2, 3.
                  -p :3 6:12:3 -> Processos: 1, 2, 3, 6, 9, 12                                                                  
      """
    )
  )

  # Adiciona ao grupo de opções gerais, referenciado por opcoes, a opção
  # -t ou --threads que permite ao usuário definir os possíveis valores
  # da variável de configuração que define o números de threads por
  # processo. O uso da opção habilita o uso das configurações
  # otimizadas, e as outras opções referentes às outras variáveis de
  # configuração serão definidas em '1' se não forem definidas pelo
  # usuário.
  opcoes.add_argument(
    "-t", "--threads", 
    type=str, 
    nargs="+", 
    default=None, 
    help=textwrap.dedent(
      """\
        Lista com os possíveis números de threads, se a aplicação usa "
        "múltiplas threads por processo."
        Usada conjuntamente com as opções -n e -p, que terão os valores "
        "padrão se não forem usadas."
        Cada elemento da lista está no formato i:e:s, onde i é o número "
        "inicial, f é o final e s é o passo."
        Pode-se omitir o i, que será igual a 1, o e, que será igual a i, e o "
        "s, que será igual a 1."
        Default 1:1.

        Exemplos: -t 1 2:24:2 -> Threads: 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, "
        "20, 22, 24"
                  -t 2 :24:8  -> Threads: 2, 24, 32, 40, 48.
                  -t 2 24 48  -> Threads: 2, 24, 48.  
      """
    )
  )

  # Adiciona ao grupo de opções gerais, referenciado por opcoes, a opção
  # -v ou --verbose que habilita a verbosidade do script de otimização,
  # mostrando diversas informações sobre o processo de escolha da melhor
  # sugestão de configuração.
  opcoes.add_argument(
    "-v", "--verbose", 
    action="store_true", 
    default=False, 
    help="Habilita a verbosidade do script."
  )

  # Adiciona ao grupo de opções gerais, referenciado por opcoes, a opção
  # -l ou --list que lista todas as aplicações que podem ter as suas
  # execuções otimizadas pelo script de otimização. A lista mostra a
  # aplicação com os possíveis nomes que podem ser usados para escolher
  # a aplivação na linha de comando.
  opcoes.add_argument(
    "-l", "--list", 
    action="store_true", 
    default=False, 
    help="Lista as aplicações cujas execuções podem ser otimizadas pelo "
         "script."
  )

  # Adiciona ao grupo de ajuda, referenciado por ajuda, a opção -h ou
  # --help que mostra a ajuda do script de otimização, com a descrição
  # da linha de comando e quais são as opções disponíveis e como elas
  # devem ser usadas.
  ajuda.add_argument(
    "-h", "--help", 
    action="help", 
    help="Mostra esta mensagem de ajuda e sai."
  )

  # Define a string "--" usada para definir, nos parâmetros passados ao
  # script quando o usuário executou ele no terminal, o ponto em que
  # temos os parâmetros do script nos patâmetros do primeiro até o
  # parâmetro igual à string, e o que vem depois deste parâmetro, o 
  # nome da aplicação seguido pelos parâmetros da aplicação. 
  # TODO: Defini o valor fixo "--"" porque é o usado na maior parte dos
  # programas e scrips para separar os parâmetros do script dos da 
  # aplicação que o script trata,
  application_param_separator = '--'

  # Verifica a string application_param_separator que define o serapador
  # está na lista de argumntos passados ao script e dados na lista
  # sys.argv.
  if application_param_separator in sys.argv:
    # Como o separador está na lista, primeiramente descobrimos a sua
    # posição separator_pos na lista,
    separator_pos = sys.argv.index(application_param_separator)

    # Como as listas do Python são indexadas a partir da posição 0, e
    # como o primeiro parâmetro da lista sys.argv sempre será, como em
    # outros comandos de terminal do Linux e suas variantes, o nome do
    # script de otimização, então os parâmetros do script de otimização
    # estarão na posição de 1 até separator_pos - 1. A lista com esses
    # parâmetros será armazenada em script_args.
    script_args = sys.argv[1:separator_pos]

    # Já o nome da aplicação e os parâmetros da aplicação estarão na
    # posição de separator_pos + 1 até o final da lista. A lista com
    # o nome da plicação e os seus parâmetros será armazenada em 
    # application_args. Note que a sintaxe do script de otimização exige
    # que o primeiro elemento dessa lista seja o nome da aplicação. 
    application_args = sys.argv[separator_pos+1:]

    # Apesar de o usuário ter usado o separador, ainda precisamos 
    # verificar se ele passou um nome da aplicação e pelo menos um
    # parâmetro da aplicação, pois precisamos de no mínimo um parâmetro
    # da aplicação para treinar o preditor para a aplicação;
  else:
    # Se o separador não estiver na lista, então todos os parâmetros
    # serão atribuídos a lista de parâmetros do script de otimização.
    # TODO: Eu preciso verificar se é necessário usar a função do 
    # analisador da linha de comando para processar a linha de comandos
    # neste caso, porque o script não irá executar já que nção foi
    # definido o nome e nem os parâmetros da aplicação.
    script_args = sys.argv[1:]

    # Como não foi usado o separador, então não foi passado o nome da
    # aplicação e nem a lisra de argumentos. 
    application_args = None

  # Processa os parâmetros passados ao script de otimização, que inclui
  # os parâmtros relacionados ao script de submissão baseado na melhor
  # sugestão de configuração, e outros parâmetros, como o para listar as
  # aplicações que podem ser otimizadas.
  user_args = parser.parse_args(script_args)

  # Verifica se o usuário não passou um parâmetro que não necessita do
  # nome da aplicação e seus parâmetros. No momento, a opção de listar
  # as aplicações que podem ser otimiadas é a única opção que não 
  # precisa do nome da aplicação.
  ignore_application_name = not user_args.list

  # Se application_args for None, então o usuário tentou executar o 
  # script sem usar o separador "--" ou não passou pelo menos o nome da
  # aplicação e um parâmetro da aplicação depois dele. Então, precisamos
  # informar o erro e mostrar a ajuda do script de otimização, que pode
  # ser mostrada pela função print_help do analisador da linha de
  # comando parser.
  if application_args is None and ignore_application_name:
      print("Não foi definido o nome da aplicação e pelo menos um argumento "
            "da aplicação!")
      parser.print_help()

  # Retorna a tupla composta pela objeto NameSpace com os parâmetros do
  # scriptr de otimização, se o usuário usou o separador da aplicação
  # e passou pelo menos dois parâmetros, pois precisamos usar pelo menos
  # um parâmetro da aplicação, ou None se os parâmetros não foram
  # passados ou somente foi passado o nome da aplicação, e uma
  # referência para o objeto parser.
  return user_args, application_args

def get_type(type_name):
  """
  Função para retornar a classe do Python associada a um tipo definido 
  por uma string e retornar uma referência para a classe correspondente
  do Python. Esta string é usada ao definir os tipos dos parâmetros da
  aplicação que os usuários precisam necessariamente passar ao otimizar
  a aplicação. No momento, são três tipos identificados pela string dada
  como parâmetro:

  "integer": o tipo é um inteiro, então é retornada uma referNeciia para
  a classe int do Python.
  "floating-point": o tipo é um número de ponto flutuante, então é
  retornada uma referNeciia para a classe float do Python.
  "string": o tipo é uma string, então é retornada uma referNeciia para
  a classe str do Python.

  
  Parâmetros:
      type_name (str): Nome do tipo, sendo "integer" para inteiros, 
                       "floating-point" para npumeros de ponto flutuante
                       ou "string" para uma string. Se o tipo for
                       inválido, o tipo é considerado como string.
                   
  Retorna: 
    <class int>: Se a string dada em type_name for "integer".
    <class 'float'>: Se a string dada em type_name for "floating-point".
    <class 'str'>: Se a string dada em type_name for "string" ou
                   inválida.    
  """

  # Dicionáro auxiliar que mapeia cada tipo, identificado pela string
  # que define uma chave do dicionário, e o valor desta chave, que é uma
  # referência para a classe do tipo correspondente à chave em Python.
  # 
  # - A chave "integer" tem uma referência para a classe int.
  # - A chave "floating-point" tem uma referência para a classe float.
  # - A chave "string" tem uma referência para a classe str.
  types_map = {
    'integer': int,
    'floating-point': float,
    'string': str
  }

  # Usa a função get do dicionário para retornar a referência para a 
  # classe identificada pela chave type_name, se esta chave existir, ou
  # o valor default com a referência para a classe str se a chave (ou
  # seja, o tipo dado em type_name) não existir.
  return types_map.get(type_name, str)    

def get_options_suggestion(suggestion_args):
  """
  Função para converter uma das componentes de uma configuração, ou
  seja, uma das variáveis de configuração como o número de nós, de
  processos por nó, ou de threads por processo. Para cada opção, o
  formato geral da opção é uma lista de strings, separados por espaços,
  no formato início:fim:incremento. Em cada string, pode ser otimido
  cada um dos componentes. Se o início for otimido, ele será 1. Se o fim
  for omitido, ele será igual ao início, e se o incremento for otimido,
  ele será igual a 1. Exemplos:

  2:12:2 -> 2, 4, 6, 8, 10 e 12.
  :4: -> 1, 2, 3 e 4.
  4::2 12:24:4 -> 4, 12, 16, 20 e 24. 
  :: 6:10: -> 1, 6, 7, 8, 9 e 10

  Parâmetros:
    suggestion_args (list[str] | None): Se for uma lista de strings com
                                        os parâmetros, sendo que cada
                                        string precisa estar no formato
                                        dado anteriormente. Se for None, 
                                        processa uma lista com uma única
                                        string '1:1'.

  Retorna: 
    list[int] | None: Se todas as strings da lista foram processadas 
                      corretamente, ou seja ínicio, fim e incremento
                      eram números inteiros, retorna uma lista de
                      inteiros com todos os possíveis valores diferentes
                      definidos pelas strings da lista, como mostrado
                      nos exemplos anteriores, ou None, se algum erro
                      ocorrer ao processar uma das strings.
  """

  # Inicializa a lista que armazenará os valores definidos pelas strings
  # da passadas na lista suggestion_args, com uma lista vazia, pois
  # ainda não processamos nenhuma string da lista suggestion_args.
  options = []

  # Se suggestion_args for None, então será processada uma lista fixa.
  if suggestion_args is None:
    # A lista fixa será composta por uma única string '1:1'
    # representando a lista de inteiros composta somente pelo inteiro 1.
    suggestion_args = ['1:1']

  # Avalia cada string da lista suggestion_args, sendo que a string a
  # ser avaliada é referenciada pela variável args.
  for arg in suggestion_args:
    # Tenta processar a lista, gerando uma exceção, que será capturada
    # se, ao  converter ínicio, fim ou incremento da string
    # "início:fim:incremento", se existirem, a string nçao puder ser
    # convertida para um número inteiro.
    try:
      # Como vimos, o formato de cada string da lista pode ser 
      # ínicio:fim:invremento, e não existir início, início = 1, se não
      # existir fim, fim = início e se não existir incremento,
      # incremento = 1.

      # Primeiramente verificamos se a string tem pelo menos um ":".
      if ':' in arg:
        # Se existir pelo menos um ":", usamos a função split do objeto
        # str do Python para dividir a string em arqs em várias strings,
        # usando o ":" como o separador das strings. A função retorna
        # uma lista com as strings definidas pela divisão nos
        # separadores ":", cuja referência será armazenada em params.
        # Como temos pelo menos um ":", então a string poderá ter ou não
        # o incremento. Em qualquer caso, o incremento sera o terceiro
        # parâmetro da lista params.
        params = arg.split(':')

        # O primeiro parâmetro sempre será ínicio, mesmo quando não for
        # definido, pois neste caso a string seria vazia, já que não tem
        # nenhum caractere antes do primeiro ":". Este parâmetro será a
        # string que começa na posição inicial (0) de args até a posição
        # anterior a do primeiro ":". A função strip do objeto str é
        # usada para remover eventuais espaços que existam entre o
        # possível inteiro que vamos posteriormente tentar converter.
        strstart = params[0].strip()

        # Verifica se o tamanho da string strstart é maior do que 0.
        if len(strstart) > 0:
          # Se o tamanho for maior do que 0 (o que ocorrerá se não
          # existirem separadores e arg não for uma string vazia ou se
          # existir pelo menos um separador e ele não for o primeiro
          # caractere de arg), tenta converter strstart para inteiro e
          # armazena o valor em start.
          start = int(strstart) 
        else:
          # Se o tamanho da string strstart for 0, então define o valor
          # de start para 1.
          start = 1   

        # O segundo parâmetro sempre será fim, mesmo quando não for
        # definido, pois neste caso a string seria vazia, já que não tem
        # nenhum caractere depois do primeiro ":" e o final da string,
        # se não existir um segundo ":", ou seja, incremento não foi
        # definido, ou entre o primeiro e o segundo ":". Este parâmetro
        # será a string que começa na posição após a posição do primeiro
        # ":" até o caractere final, se o segundo ":" não existie, ou a
        # strings entre os dois caracteres ":" em caso contrário. A
        # função strip do objeto str é usada para remover eventuais
        # espaços que existam entre o possível inteiro que vamos 
        # posteriormente tentar converter.
        strend = params[1].strip()

        # Verifica se o tamanho da string strend é maior do que 0.
        if len(strend) > 0:
          # Se o tamanho for maior do que 0 (o que ocorrerá se existir
          # somente um separador e exsitirem caracteres depois dele ou
          # dois separadores e existirem caracteres entre estes
          # separadores), tenta converter strend para inteiro e armazena
          # o valor em end.
          end = int(strend) 
        else:
          # Se o tamanho da string strend for 0, então define o valor de
          # end igual ao de start convertido anteriormente.
          end = start

        # Verificamos se existem pelo menos dois separadores ":", o que
        # implica que a lista parms terá duas ou mais strings.
        if len(params) > 2:
          # Se existirem pelo menos três strings em params, então temos
          # pelo menos dois separadores ":" e o incremento será a
          # terceira string de params. Se separadores adicionais forem
          # usados, no momento eles serão ignorados. O incremento será a
          # string depois do segundo separador até o final, se não
          # existir um terceiro separador (que não seria correto), ou
          # até antes deste separador, em caso contrário. Novamente a
          # string pode ser vazia se não existirem caracteres definidos
          # por uma das strings descritas anteriormente.
          # TODO: Será que deveríamos dar um erro caso len(params) for
          # maior do que 3?
          strstep = params[2].strip()
        else:
          # Se params tiver menos do que 3 strings, então o incremento
          # não foi definido porque somente foi usado um ":", logo o
          # incremento será definido com o inteiro 1.
          strstep = "1"

        # Verifica se o tamanho da string strstep é maior do que 0.
        if len(strstep) > 0:
          # Se o tamanho for maior do que 0 (o que ocorrerá se dois ou
          # mais separadores foram definidos e existia uma string não
          # vazia, ou um ou nenum separador foi definido e a string foi
          # definida para "1"), tenta converter strstep para inteiro e
          # armazena o valor em step.
          step = int(strstep) 
        else:
          # Se o tamanho da string strstep for 0, então define o valor
          # de step para 1.
          step = 1

        # Uma vez definidos start, end e step, usamos a função range do
        # Python para gerar os valores entre start e end, com
        # incrementos dados em step, lembrando que, como o range não
        # inclui o ultimo valor, devemos então passar end+1. Depois,
        # usamos a função extend da lista do Python para inserir todos
        # os inteiros definidos entre start e end, inclusos, em
        # incrementos step na lista options com os valores definidos
        # pelas strings em suggestion_args. 
        options.extend(range(start, end+1, step))  
      else:
        # Se não existirem separadores ":" na string arg, então ela é
        # composta por uma string que deveria ser início e ser um número
        # inteiro, logo tentaremos convertar arq para inteiro e, se
        # nenuma exceção ocorrer, armazenamos o valor convertido em
        # param.
        param = int(arg)

        # Como arq tinha somente um inteiro que foi convertido de string
        # para inteiro e armazenado em param, adicionamos este inteiro a
        # lista options com os valores definidos pelas strings em
        # suggestion_args. 
        options.append(param)

    # Verifica se alguma exceção ocorreu ao fazer uma das conversões
    # para inteiro descritas anteriormemte.    
    except Exception as e:
      # Se ocorrer alguma exceção, indicando um erro ao tentar converter
      # os inteiros, é retornado None ao invés da lista.    
      return None
  
  # Como podem ser gerados valores duplicados ao processar as strngs em
  # suggestion_args, usamos o troque de converter a lista em conjunto
  # para remover os valores dulicados e depois geramos uma lista com
  # esses valores ordenados usando a função sorted.
  options = sorted(list(set(options)))

  # Retorna a lista dos valores deinidos pelas strins em suggestion_args
  # se nenhum erro ocorreu, ou None se algum erro ocorreu ao processar
  # uma das strings em suggestion_args.
  return options

def convert_user_params(required_applicaion_params, conversions, 
                        application_configs_dir):
  """
  Função para comverter um ou mais parâmetros da aplicação passados
  diretamente pelo usuário, em um ou mais parâmetros da aplicação que
  efetivamente foram usados ao treinar os modelos. No momento, temos as
  seguintes operações, definidas no dicionário conversions, sendo que a
  chave define a operação:

  "copy": O parâmetro da apluicação definido pelo usuário é usado
          diretamente ao treinar os modelos, e será copiado para o
          parâmetro da aplicação correspondente usado ao treinar o
          modelo, que pode ter ou não o mesmo nome do parâmetro da
          aplicação fornecido pelo usuário, mas sempre usará o mesmo
          parâmetro passado diretamente pelo usuário.
  "filesize": Supõe que o usuário passou, como um dos parâmetros da
              aplicação, um nome do arquivo, e que foi efetivamente
              usado, ao treinar os modelos, o tamanho deste arquivo no
              disco.
  "map": Define que um parâmetro da aplicação, usado ao treinar os
         modelos, é definido a partir de uma tabela no formato csv, em
         que cada linha define os possíveis valores para os parâmetros
         da aplicação definidos pela operação de mapeamento, sendo que
         cada parâmetro obrigatoriamente terá uma coluna nessa tabela, e
         também uma coluna que define, para este parâmetro o seu valor
         considerando os valores dos parâmetros da aplicação definidos
         pelos valores das colunas, associadas aos parâmetros da
         aplicação usados no mapeamento, desta linha. Por exemplo, no
         benchmark NPB, supondo um caso de teste hipotético em que o
         usuário escolheria um dos benchmarks usados ao treinar os
         modelos, bt-mz, lu-mz e sp-mz, e uma das classes também usadas
         ao treinar os modelos, A, B, C ou D, que existirá um
         mapeamento, no caso em uma mesma tabela armazenada em um
         arquivo .csv, para cada parâmetro de aplicação efetivamente
         usado ao treinar os modelos, ou seja, Zone X, Zone Y,
         iteration, Grid X, Grid Y e GridZ. Nete arquivo .csv, que será
         lido e convertido em um DataFrame do Pandas, existirá uma linha
         para cada combinação de benchmark e classe, uma coluna para o
         benchmark, classe e cada uma das variáveis Zone X, Zone Y,
         iteration, Grid X, Grid Y e GridZ usadas ao treinar os modelos,
         sendo que uma linha para uma combinação de benchmark e classe
         definirá o valor de cada uma das variáveis usadas ao treinar os
         modelos, como foi definido na tabela com as relação entre
         benchmark, classe, Zone X, Zone Y, iteration, Grid X, Grid Y e
         GridZ dada no artigo estendido do SSCAD.

  Parâmetros:
    required_applicaion_params (NameSpace): Objeto com todos os
                                            parâmetros da aplicaçaõ e
                                            os seus valores, passados
                                            pelo usuário como parâmetros
                                            da aplucação na linha de
                                            comando do script de
                                            otimização. Os parâmetros
                                            são obtidos usando um objeto
                                            da classe 
                                            argparse.ArgumentParser do
                                            Python, a mesma usadas para
                                            processar os parâmetros do
                                            script de otimização.       
    conversions: (dict[tuple]): lista com as conversões a serem feitas, 
                                baseadas nas variáveis de aplicação
                                definidas em required_applicaion_params,
                                sendo cada chave o nome da variável da
                                aplicação usada ao treinar os modelos, e
                                a tupla associada a essa chave definindo
                                a conversão usada para obter o valor
                                dessa variável. O primeiro elemento de
                                uma dessas tuplas é o nome da conversão,
                                como definido anteriormente. Os demais
                                elementos dependerão da conversão. Para
                                as conversões "copy" e "filesize", o
                                segundo elemento é o nome da variável de
                                aplicação usada na conversão com o valor
                                passado pelo usuário definido em
                                required_applicaion_params. Já para a
                                operação "map", o segundo até o
                                pénúltimo elemento são as variáveis
                                usadas no mapeamento, também definidas
                                pelos usuários em
                                required_applicaion_params, usadas ao
                                gerar o valor da aplicação associado à
                                chave que referência a tupla, e o último
                                elemento é o nome do arquivo .csv com a
                                tabela com os  mapeamentos. 
  application_configs_dir (Path): Diretório com os arquivos de
                                  configuração,  que é necessário porque
                                  todas as tabelas de conversão usadas
                                  pela operação "map" serão armazenadas.

  Retorna: 
    dict[int | float] | None: Se nenhum erro ocorrer durante as
                              conversões, retorna um dicionário que
                              define, para cada variável da aplicação
                              usadas nos treinamentos e definida por
                              uma das chaves no dicionário conversions
                              com todas as variáveis de aplicação a
                              serem convertidas e usadas nos
                              treinamentos, o seu valor após a
                              converção, que poderá ser um inteiro ou um
                              número de ponto flutuante dependo da
                              variável da aplicação. Porém, se algum
                              erro de conversão ocorrer, retorna None
                              para informar que pelo menos uma conversão
                              não foi feita.
  """

  def copy_func(*args):
    """
      Função para simplesmente copiar o valor de uma variável de
      aplicação definida pelo usuário. A tupla, neste caso, terá somente
      uma componente que será o nome da variável definida pelo usuário
      da qual deveremos copiar o valor.

      Parâmetros:
        args (tuple): tupla com somente um elemento, uma string com o
                      nome da variável de aplicação, passada pelo
                      usuário, a ter o seu valor copiado.

      Retorna: 
        int | float: Como a conversão é de cópia, retorna diretamente o
                     valor da variável definido pelo usuário do script
                     de otimização, que pode ser um int ou um float, já
                     que essa variável será usada ao fazer predições em
                     um modelos de regressão.
    """
    # Como todas as variáveis da aplicação definidas pelo usuário estão
    # armazemadas na tupla nomeada required_applicaion_params, então
    # verificamos se essa tupla tem um campo com o nome dado em args[0].
    # Caso o nome não exista, que seria um erro no código do script de
    # otimização, a exceção "KeyError" é gerada pela função getattr e
    # capturada pelo bloco try que usa a função copy_func, e o erro
    # informado ao usuário, para ser reportado a equipe de suporte do
    # script de otimização. Se o nome da variável de aplicação for
    # válido, getattr retorna o valor desta variável.
    return getattr(required_applicaion_params, args[0])

  def filesize_func(*args):
    """
      Função para obter o tamanho do arquio definido por uma variável de 
      aplicação definida pelo usuário. A tupla, neste caso, terá somente
      uma componente que será o nome da variável definida pelo usuário
      que conterá o caminho completo do arquivo para o qual desejamos
      obter o tamanho. 

      Parâmetros:
        args (tuple): tupla com somente um elemento, uma string com o
                      nome da variável de aplicação, passada pelo
                      usuário, com o caminho completo do nome do
                      arquivo, dado pelo usuário, para o qual vamos
                      calcular o tamanho e usar como uma das variáveis
                      de aplicação ao fazer as prediçoes nos modelos.

      Retorna: 
        int | None: Se o caminho do arquivo dado no único elemento da
                    tupla for válido, ou seja, é o caminho de um arquivo
                    que existe e está acessível pelo usuário, retorna o
                    tamanho desse arquivo em bytes. Em caso contrário,
                    ou seja, não foi possível obter o tamanho devido ao
                    caminho ser de um arquivo inexistente, um diretório
                    ou algum outro tipo de objeto do sistema de
                    arquivos, retorna None.
    """
    # Como todas as variáveis da aplicação definidas pelo usuário estão
    # armazemadas na tupla nomeada required_applicaion_params então,
    # como antes, a função getattr gerará uma exceção "KeyError" se
    # args[0] não existir na tupla. Neste caso, o valor retornado por
    # getattr será o caminho completo do arquivo para o qual desejamos
    # obter o tamanho, passado pelo usuário. Geramos uma instância do
    # objeto Path do Pandas para poder acessar o arquivo, e armazena a
    # referência para essa instância em file_path.
    file_path = Path(getattr(required_applicaion_params, args[0]))

    # Verificamos, usando a função is_file da classe Path, se o caminho
    # passado pelo o usuário é de um arquivo.
    if file_path.is_file():
      # Se o caminho for de um arquivo válido, ou seja, é realmente um
      # arquivo que existe no sistema de arquivos e pode ser acessado,
      # a função stat da classe Path é usada para retornar, em uma tupla
      # nomeada, as  propriedades do arquivo cujo caminho foi associado
      # ao objeto referenciado por file_path. Um dos campos dessa tupla
      # é o st_size que armazena exatamente o tamanho do arquivo que
      # desejamos. Logo, retornamos o valor desse campo pois será o
      # tamanho do arquivo. 
      return file_path.stat().st_size
    elif file_path.is_dir():
      # Se o caminhio for um diretório, o usuário definiu incorretamente
      # o parâmertro da aplicação para o qual desejavamos obter o
      # tamanho do arquivo, pois este caminho deveria ser para um
      # arquivo. Neste caso, mostramos uma mensagem de erro com o
      # caminho completo do arquivo e retornamos None para indicar que
      # ocorreu um erro ao converter a variável da aplicação passada
      # pelo usuário.
      print(f"❌ Erro ao converter a variável {args[0]}! O caminho "
            f"{file_path.resolve()} é um diretório!")

      # Como não podemos obter o tamanho de um diretório, retornamos
      # None.
      return None
    elif file_path.exists():
      # Se o caminhio não for um arquivo e nem um diretório, também o
      # usuário definiu incorretamente o parâmertro da aplicação para o
      # qual desejavamos obter o tamanho do arquivo, pois este caminho
      # referência um arquivo que somente pode ser um arquivo especial, 
      # pois a função exists retornou true e as funções is_file e is_dir
      # retornaram false. Neste caso, mostramos uma mensagem de erro com
      # o caminho completo do arquivo o retornamos None para indicar que
      # ocorreu um erro ao converter a variável da aplicação passada
      # pelo usuário.
      print(f"❌ Erro ao converter a variável {args[0]}! O caminho "
            f"{file_path.resolve()} é um arquivo especial do sistema "
            "oparacional!")
      # Como não podemos obter o tamanho de um objeto do sistema de
      # arquivos com um tipo desconhecido ou que efetivamente não
      # existe, retornamos None.
      return None
    else:
      # Este caso é quando o caminho é inválido, também o usuário
      # definiu incorretamente o parâmertro da aplicação para o qual
      # desejavamos obter o tamanho do arquivo, pois este caminho não
      # existe. Neste caso, mostramos uma mensagem de erro com o caminho
      # completo do arquivo, eretornamos None.
      print(f"❌ Erro ao converter a variável {args[0]}! O caminho "
            f"{file_path.resolve()} é inválido, pois não existe ou o "
            "usuário não tem permissão para acessar o caminho!")

      # Como não podemos obter o tamanho de um objeto do sistema de
      # arquivos com um tipo desconhecido ou que efetivamente não
      # existe, retornamos None.
      return None

  def map_func(user_arg_name, *args):
    """
    Função que mapeia um conjunto de variáveis de entrada, passadas na
    tupla, que define o arquivo com a tabela csv com os mapeamentos e as
    variáveis de aplicação, definidas pelo usuário, usadas
    nestemapeamento e a variável que será gerada pelo mapeamento porque,
    ao contrário das outras operações, precisamos saber do nome da
    variável de destino para acessar a coluna correta com os valores do
    mapeamento dessa variável.

    Parâmetros:
         user_arg_name: Nome da variável que será mapeada, ou seja,
                        para a qual iremos usar a tabela de mapeamento
                        para obter o seu valor, de acordo com os valores
                        das variáveis de aplicação cujos nomes são
                        passados a partir do segundo elemento da tupla
                        args definida a segir. O nome da variável
                        precisa ser fornecido para saber qual coluna da
                        tabela de mapeamento deveremos considerar.
         args (tuple): Tupla em que o primeiro elemento é o nome do
                       arquivo com a tabela com os mapeamentos das
                       variáveis da aplicação definidas pelo usuário nas
                       variáveis de aplicação efetivamente usadas ao
                       treinar os modelos, e as strings com os nomes das
                       variáveis de aplicação definidas pelo usuário
                       cujos valores serão usados ao fazer o mepamento.
 
       Retorna: 
          int | float | None: O valor mapeado para a variável de
                              aplicação, efetivamente usada para terinar
                              os modelos, cujo nome foi dado em
                              user_arg_name, se nenhum erro ocorrer
                              durante o mapeamento. O tipo no caso de
                              não existirem erros dependerá do valor
                              definido para a variável user_arg_name na
                              tabela de mapeamento. Se algum erro
                              ocorrer durante o mapeamento, retorna
                              None.
  
    """
 
    # O primeiro elemento da tupla args é o nome do arquivo com a tabela
    # no formato CSV com os mapeamentos, para o qual a sua referência é
    # também armazenada na variável dataframe_map_file_name. O nome dado
    # no primeiro elemento é somente o nome do arquivo com a tabela,
    # sem o caminho do diretório com os arquivos de configurão das
    # aplicações.
    dataframe_map_file_name = args[0]

    # Como o primeiro elemento da tupla é somente o nome do arquivo com
    # a tabela precisamos obter o caminho completo para ler a tabela,
    # que será o caminho com os arquivos de configuração das aplicações
    # dado em application_configs_dir mais uma "/" mais o nome do
    # arquivo porque todos as tabelas de mapeamento estão nesse
    # diretório com os arquivos de configuração das aplicaações.
    dataframe_map_full_path_name = (Path(application_configs_dir) 
                                    / dataframe_map_file_name)

    # Verifica se a chave com o nome do arquivo dataframe_map_file_name
    # existe no dicionário dataframe_map_dict.
    if dataframe_map_file_name in dataframe_map_dict.keys():
      # Se existe uma chave com o nome do arquivo armazenado em 
      # dataframe_map_file_name, então basta usarmos o dataframe dado 
      # pela referência desta chave ao fazer o mapeamento da variável
      # user_arg_name. A referência para o dataframe jpa lido é copiada
      # em df_map.
      df_map = dataframe_map_dict[dataframe_map_file_name]
    else:
      # Se a chave dataframe_map_file_name com o nome do arquivo com a
      # tabela de mapeamento não existir no dicionário 
      # dataframe_map_dict, usa a função read_csv do pandas para ler a
      # tabela no formato CSV e criar um objeto DataFrame do Pandas com
      # a tabela lida. Uma referência ao dataframe lido é colocada em
      # df_map.
      df_map = pd.read_csv(dataframe_map_full_path_name) 

      # TODO: Deixei esta depuração, habilitada pela variável de
      # ambiente APPOPTIMIZER_DEBUG, para mostrar a tabela de mapeamento
      # lida. Podemos tirar todas as depurações no futuro. 
      # Imprime a tabela no formato CSV com os mapeamentos lida do
      # arquivo dataframe_map_file_name e convertida para um DataFrame
      # do Pandas, mostrando o nome do arquivo dataframe_map_file_name e
      # a tabela lida, usando a referência ao dataframe com a tabela
      # colocada em df_map. 
      if debug_code:
        print(f"➡️  Dataframe de mapeamento {dataframe_map_file_name}: \n\n")
        print(df_map.to_markdown(tablefmt="grid"))

      # Cria uma chave com o nome dataframe_map_file_name do arquivo com
      # a tabela de mapeamento lida e associa esta chave à referência ao
      # objeto, com o dataframe com a tabela lida, dada em df_map.
      dataframe_map_dict[dataframe_map_file_name] = df_map

    # Cria uma lista vazia a partir da qual será montada a condicional
    # para buscar, no dataframe com os mapeamentos dado em df_map, as
    # colunas com os valores das variáveis de aplicção definidas pelo
    # elemento sdo segundo (posição 1 de args) até o último elemento em
    # args. A condional será uma string para procurar a linha correta,
    # com os valores das variáveis dadas a partir da segunda posição da
    # tupla, adequada para ser usada pela função query do Pandas, sendo
    # que cda elemento da lista será uma das condições a serem buscadas.
    list_search = []

    # Cria a parte da condiconal para cada variável da aplicação dada a
    # partir do segudo elemento da tupla args.
    for user_option in args[1:]:
      # Obtém o valor associado a opção dada pelo elemento atual
      # avaliado, que deveria ser o nome de uma das variáveis em
      # required_applicaion_params. Se nenhum erro ocorrer, criamos a
      # condição "user_option == valor", em que o valor é o valor para a
      # variável de aplicação, definida pelo usuário, dados pelo
      # atributo user_option da tupla nomeada
      # required_applicaion_params, retornado pela função getattr. Se o
      # nome da variável de aplicação não existir, getattr gerará uma
      # exceção KeyError. Depois de criada a string com a condição, ela
      # é adicionada a lista list_search com todas as condições que
      # devem ser verificadas (como estaos procurando as linhas das
      # variáveis de aplicação, definidas pelo usuário, dadas a partir
      # do segundo elemento da tupla arqs, existirá uma condição para
      # cada um desses elementos).
      list_search.append(
          f"{user_option}.astype('str') == "
          f"'{getattr(required_applicaion_params, user_option)}'"
      )

    # Cria a string de condição, usando a operação join com a string 
    # " and ", implicando que a string final conterá todos os elementos
    # da lista list_search, que são cada uma das condições necessárias
    # para encontrar a linha correta com o mapeamento, pela operação
    # "and" que indicará na condição final que todas as condições
    # definidas pelas strings da lista list_search deverão ser
    # verdadeiras.
    str_search = ' and '.join(list_search)

    # Usa a função query do Pandas para descobrir a linha, que deverá
    # ser única do dataframe com os valores das variáveis de aplicação 
    # definidas a partir do segundo elemento da tupla args, sendo os
    # valores definidos pelo usuário para essas variáveis de aplicação
    # dados na tupla nomeada required_applicaion_params. A refeência ao
    # dataframe resultante da busca feita em query é armazenado em
    # result_df.
    result_df = df_map.query(str_search).reset_index(drop=True)

    # Se o dataframe armazenado em result_df estiver vazio, então a
    # tabela de mapeamento não tem uma linha para mapear todos os
    # parâmetros da aplicação defidos a partir do segundo elemento da
    # tupla args, definidos na tupla nomeada required_applicaion_params.
    if result_df.empty:
      # Se a busca não apresentou resultados, vamos criar uma mensagem
      # informando o problema de não ser possível fazer o mapeamemto, o
      # que não deveria ocorrer. A não existẽncia de um mapeamento
      # indica que a tabela com os mapeamentos está incompleta, e isso
      # precisa ser reportado ao suporte resposável pelo scipt de
      # otimização.

      # Lista com os erros que ocorreram, ou seja, os valores das
      # variáveis em args para os quais os valores, dados em
      # required_applicaion_params, não existem na tabela de mapeamento,
      # o que impossibilitou o mapeamento de ser feito.
      list_erros = []

      # Vamos verificar para quais variáveis da aplicação cujos nomes
      # foram dados a partir do segundo elemento da tupla args, para
      # qual(is) dessa(s) variáveis não existe o valor da variável
      # definido, pelo usuário, na tupla required_applicaion_params.
      for user_option in args[1:]:
        # Obtem o valor da variável de aplicação atualmente avaliada 
        # user_option a partir da tupla nomeada
        # required_applicaion_params, usado a função getattr do Python.
        # Novamente, um nome errado irá gerar a exceção KeyErro que será
        # capturada e o erro reportado.
        option_value = getattr(required_applicaion_params, user_option)

        # Verifica se o valor não está na coluna associdada à variável
        # de apliçação com o nome user_options na tabela de mapeamentos
        # df_map, pois se o valor não exsitir, então a variável
        # user_option foi uma das responsáveis pela falha em encontrar
        # um mapeamento.
        if option_value not in df_map[user_option].values:
          # Adiciona a variável da aplicação que impediu o mapeamento,
          # junto com o valor dessa variável definido pelo usuário. 
          list_erros.append(f"{user_option} = {option_value}")

      # Imprime a lista com todas as variávels. Separei a impressão de
      # acordo com o tamanho da lista para evitar usar (s), (es) ou
      # (ões) em todas as palavras da frase indicando o erro.
      if len(list_erros) == 1:    
        # Frase quando somente uma variável da aplucação impediu o
        # mapeamento.
        print(f"❌ Valor inválido dado para a opção: {list_erros[0]}")
      else:                
        # Frase quando mais de uma variável da aplucação impediu o
        # mapeamento.
        print(f"❌ Valores inválidos dados para as opções: "
              "{', '.join(list_erros)}")
  
      # Como não podemos fazer o mapeamento devido à não existirem
      # mapeamentos para todas as variáveis de aplicação definidos pelo
      # usuário em required_applicaion_params na tabela mapeamento,
      # retornamos None.
      mapped_value = None
    elif len(result_df) == 1:
      # Para o mapeamento ser correto, não deveria, na tabela de
      # mapeamentos, existir dois mapeamentos para um mesmo conjunto de
      # valores das variáveis da aplicação definidas em
      # required_applicaion_params, considerando as variáveis desta
      # tupla nomeada definidas a partir do segundo elemento de args.
      mapped_value = result_df.loc[0, user_arg_name]  

      # Como encontramos um unico mapeamento, retornamos o seu valor.
      return mapped_value
    else:
      # A tabela result_df tem mais de uma linha, o que significa que
      # temos mais de um possível mapeamento, o que não deveria ocorrer,
      # pois o mapeamento deve ser único. Neste caso, indicamos o erro
      # para o usuário, imprimindo o resultado que informa que existem
      # dois ou mais mapeamentos.
      print(f"❌ A tabela de mapeamento {dataframe_map_file_name} está "
            "definindo mais de um mapeamento para as variáveis da aplicação "
            "passados pelo usuário, como indicado na tabela a seguir")
      print("\n", result_df.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", 
            sep="")
      print(f"❌ Por favor, reporte este erro ao adminstrador do sistema!")

      # Como não podemos fazer o mapeamento devido à ambiguidade,
      # retornamos None.
      return None

  # Agora vamos começar o código da função principal, que usará as
  # outras funções auxiliares definidas anteriormente

  # Dicionário com os dataframes usados pela conversão "map" que faz um 
  # mapeamento, para garantir que eles sejam lidos uma única vez se
  # forem usados em mais de um mapeamento. Neste dicionário, cada chave
  # é o nome do arquivo que contém um dataframe usado para mapear um
  # conjunto de parâmetros da aplicação, definidos pelos usuários, nos
  # parâmetros da aplicação correspondentes usados ao fazer as predições
  # nos modelos. Um exemplo é o benchmark de teste NPB do NAS. Se não
  # fosse um benchmark, o usuário poderia fornecer um das aplicações do
  # benckmark consideradas ao treinar os modelos, ou seja, bt-mz, lu-mz
  # e sp-mz, e uma das classes usadas ao treinar os modelos, A, B, C ou
  # D. Neste caso, o dataframe mapeará cada combinação de benckmark e
  # classe nos parametros relevantes sobre as aplicações e que foram
  # considerados ao treinar os modelos, Zone X, Zone Y, Iterations, 
  # Grid X, Grid Y e Grid Z, como mostrado na tabela do nosso artigo
  # estendido do SSCAD.
  dataframe_map_dict = {}

  # Usamos o block try-except para capturar eventuais erros, como um
  # arquivo inexistente ou uma chave inválida, que possam ser gerados
  # por um possível erro no código. O código final nunca deveria geare
  # esses erros, pois isso indicaria um erro no código do script de
  # otimização, já que possíveis erros do usuário nos valores dos
  # parâmetros da aplicação que ele definiu deveriam ser detectados
  # pelas funções de conversão cujos códigos foram descritos
  # anteriormente. Todos os erros, se ocorrerem, são reportados
  # informando aos usuários para entrarem em contado com o suporte
  # repassando a mensagem de erro gerada.
  try:
    # Cria o dicionário que armazenará as variáveis de aplicação
    # mapeadas a partir das variãveis de aplicações denifidas pelo
    # usuário na lista com as converões.
    converted_user_params = {}

    # Agora vamos tratar de cada uma das conversões dadas no dicionário 
    # conversions. Para isso, como conversions é um dicionário em que
    # cada chave é o nome da variável de aplicação a ser convertida a
    # partir da conversão cujos dados estão na tupla referênciada pela
    # chave, vamos então percorrer cada chave do dicionário, sendo a
    # chave para a qual faremos a conversão em cada passo armazenada em
    # variable.
    for variable in conversions:
      # Vamos agora fazer a conversão para a variável de aplicação
      # variable, usando um ou mais dos valores das variáveis da
      # aplicação definidas pelo usuário e dados na tupla conversions. 
      conversion_info = conversions[variable]

      # Como vimos na definição dessa função de conversão das variáveis
      # de aplicação, o primeiro elemento da tupla é a string que
      # identifica a conversão a ser feita, como indicado na descrição
      # no início da função.
      conversion_type = conversion_info[0]

      # Também como vimos na definição da função, os outros elementos da
      # tupla serão os parâmetros usados pela conversão definida por
      # conversion_type.
      conversion_args = conversion_info[1:]

      # Dicionário auxiilar usado para executar as funções de conversão,
      # sendo que a função partial do Python é usada para executar a
      # função quando não desejamos invocar a função no momento da sua
      # definição, no caso, da associação da função à chave que
      # identifica a conversão. Isso foi feito para facilitar a adicição
      # de novas funções de conversão que sejam necessárias nu futuro,
      # sem necessitar o uso de uma sequência if-elif-...-elif-else que
      # dependeria de quantas conversões existem, ou do uso da nova
      # estrutura match-case do Pythom que somente existe após a versão
      # 3.10 da linguagem.
      conversion_types = {
        'copy': partial(copy_func, *conversion_args),
        'filesize': partial(filesize_func, *conversion_args),
        'map':  partial(map_func, variable, *conversion_args),
      }

      # Como cada chave do dicionário conversion_types associa o nome da
      # conversão a uma função a ser executada, então usamos a chave 
      # conversion_type, com o nome da conversão, para executar a função
      # correta de conversão. Note que após acessar o diretório, usamos
      # os parênteses de chamada de uma função sem parâmetros. Este é o
      # modo de chamar uma função definida pelp partial quando não são
      # necessaŕios parâmetros adicionais além dos definidos quando o
      # partial foi usado. O valor retornado pela conversão correta é
      # armazenado em converted_value. Note que uma conversão
      # inexistente, que seria um erro no arquivo de configuração da
      # aplicação para a qual o usuário deseja otimizar, irá gerar uma
      # exceção KeyErro que será capturada e o erro reportado ao
      # usuário, recomendando que ele entre em contato com o suporte
      # responsável pelo script.
      converted_value = conversion_types[conversion_type]()

      # Se a função de conversão retornar none para uma das variáveis de
      # aplicação a serem convertidas, paramos o processo de conversão e 
      # retornamos None.
      # TODO: Talez possamos postergar isso até retornar o dicionário 
      # converted_user_params e fazer as vericações quando o dicionário
      # for retornado, pois isso permitiria descobrir mais de um erro
      # cometido pelo usuário.
      if converted_value is None:
        # Se a conversão atual não pode ser feita devido a existẽncia de
        # erros, retorna None ao invés do diconário
        # converted_user_params com as variáveis de aplicação e seus
        # valores convertidos.
        return None
      
      # Como a conversão para a variável de aplicação variable foi feita
      # com sucesso, armazena o valor converted_value desta conversão no
      # dicionário converted_user_params.
      converted_user_params[variable] = converted_value

    # Como todas as conversões foram feitas com sucesso, retorna o
    # dicionário converted_user_params com cada variável de aplicação e
    # o seu valor convertido.
    return converted_user_params

  # Um dos arquivos usados na conversão não foi encontrado. Esta exceção
  # deveria somente ocorrer ao ler uma das tabelas de mapeamento, pois a
  # conversão que retorna o tamanho do arquivo verifica se ele existe.
  except FileNotFoundError as e:
    print(f"❌ O arquivo {e.filename} nao foi encontrado.")
    print("❌ Por favor, avise o erro ao adistrador do sistema o erro: "
          f"{e.strerror}!")

    # Como oocorreu una exceção inesperada, retorna None ao invés do 
    # dicionário com os valores convertidos das variáveis de aplicação.
    return None
  # Não foi possível acessar um dos arquivos, devido a um erro de
  # permissão de acesso ao arquivo.
  except PermissionError as e:
    print(f"❌ Erro de permissão ao acessar o arquivo {e.filename}.")
    print("❌ Por favor, avise o erro ao adistrador do sistema o erro: "
          f"{e.error}.")

    # Como oocorreu una exceção inesperada, retorna None ao invés do 
    # dicionário com os valores convertidos das variáveis de aplicação.
    return None
  # Não foi possível acessar um dos arquivos, devido a um erro de
  # leitura ao acessar o arquivo.
  except IOError as e:
    print(f"❌ Erro de I/O ao ler o arquivo {e.filename}!")
    print(f"❌ Código do erro: {e.errno}; Mensagem: {e.strerror}!")
    print(f"❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como oocorreu una exceção inesperada, retorna None ao invés do 
    # dicionário com os valores convertidos das variáveis de aplicação.
    return None

  # Ocorreu um erro ao acessar uma chave de um dicionaŕio ou um campo de
  # uma tupla nomeada. Este erro não deveria ocorrer, e indica um erro
  # no código do script de otimização ou no arquivo de configuração da
  # aplicação que o usuário está tentando otimizar.
  except KeyError as e:
    print(f"❌ Erro interno ao processar o valor da opção {e.args}.")
    print(f"❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como oocorreu una exceção inesperada, retorna None ao invés do 
    # dicionário com os valores convertidos das variáveis de aplicação.
    return None

  # Ocorreu uma exceção inesperada ao tentar fazer as conversões, 
  # provavelmente devido a um erro no código do script de otimização.
  except Exception as e:
    print("❌ Erro desconhecido ao processar o valor da opção "
          f"{', '.join(e.args)}")
    print(f"❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como oocorreu una exceção inesperada, retorna None ao invés do 
    # dicionário com os valores convertidos das variáveis de aplicação.
    return None

def generate_submission_script(template_file_path, template_params):
  """
  Função para gerar o script de submissão para uma aplicação, a partir
  de um arquivo de template para a aplicação, definido pelo caminho
  template_file_path, utilizando os recursos definidos pelas valores das
  variáveis de configuração da melhor sugestão de configuração obtida
  pelo script de otimização, o tempo estimado de execução e a partição a
  ser usada para a submissão do job, que será definida a partir do tempo
  de execução e das sugestões de configuração, o uso máximo de memória e
  se a execução é ou não exclusiva. Também são passados todos os 
  parâmetros da aplicação definidos pelo usuário, que serão passados
  para o script de submissão como parâmetros da aplicação, e o nome do 
  job, que será usado para identificar o job submetido, como foi 
  definido pelo usuário na linha de comando do script de otimização. 
  A função retorna este script de submissão em uma string com o script
  de submissão.

  Parâmetros:
    template_file_path (Path): Caminho completo do arquivo de template
                               do script de submissão para a aplicação a 
                               ser otimizada, que será usado para
                               gerar o script de submissão final.
    template_params (dict): Dicionário com as variáveis de configuração
                            da melhor sugestão de configuração obtida
                            pelo script de otimização, o tempo estimado
                            de execução e a partição a ser usada para a
                            submissão do job, o nome do job e todos os
                            parâmetros da aplicação definidos pelo 
                            usuário.

  Retorna:
    str: String com o script de submissão gerado a partir do arquivo de
         template e das variáveis de configuração da melhor sugestão de
         configuração obtida pelo script de otimização, o tempo estimado
         de execução, a partição a ser usada para a submissão do job, o
         uso máximo de memória, se a execução é ou não exclusiva, e o
         nome do job e todos os parâmetros da aplicação definidos pelo
         usuário.
  """

  def format_size(size_in_bytes):
    """
    Função para formatar o tamanho de um arquivo em bytes para uma
    string com o tamanho em uma unidade mais legível, como K, M, G, T ou
    P. A função recebe o tamanho em bytes e retorna uma string com o
    tamanho formatado, arredondado para cima e com a unidade apropriada.

    Parâmetros:
      size_in_bytes (int): Tamanho do arquivo em bytes.

    Retorna:
      str: String com o tamanho do arquivo formatado, arredondado para
           cima, e com a unidade apropriada (B, K, M, G, T ou P
    """

    # Definimos a lista de labels para as unidades de medida, sendo B 
    # para bytes, K para kilobytes, M para megabytes, G para gigabytes, 
    # T para terabytes e P para petabytes. 
    labels = ['B', 'K', 'M', 'G', 'T', 'P']

    # Inicializamos o índice do label em 0, que corresponde a bytes.
    label_index = 0

    # Enquanto o tamanho em bytes for maior ou igual a 1024 e o índice
    # do label for menor que o tamanho da lista de labels menos 1, 
    # dividimos o tamanho em bytes por 1024 e incrementamos o índice do
    # label. Isso nos permite encontrar a unidade de medida apropriada
    # para o tamanho do arquivo, até o limite de petabytes, em que
    # paramos de dividir o tamanho em bytes por 1024.=, implicando que
    # para o sufixo P, podemos ter um tamanho maior que 1024 antes do
    # sufixo.
    while size_in_bytes >= 1024 and label_index < len(labels) - 1:
      # Dividimos o tamanho em bytes por 1024 para converter para a
      # próxima unidade de medida.
      size_in_bytes = size_in_bytes / 1024.0

      # Incrementamos o índice do label para usar a próxima unidade de
      # medida na lista de labels.
      label_index += 1

    # Retorna o tamanho formatado como uma string, arredondado para
    # cima e com a unidade apropriada.
    return f"{np.ceil(size_in_bytes):.0f}{labels[label_index]}"
  
  def format_time(time_in_seconds):
    """
    Função para formatar o tempo em segundos para uma string no formato
    "HH:MM:SS", se o número de dias for zero (ou seja, time_in_seconds é
    menor do que 86400 segundos), ou no formato "D-HH:MM:SS" em caso 
    contrário, sendo D o número de dias, HH o número de horas, MM o 
    número de minutos e SS o número de segundos.

    Parâmetros:
      time_in_seconds (int): Tempo em segundos. 

    Retorna:
      str: String com o tempo formatado no formato "HH:MM:SS" se o
           número de segundos for menor que 86400, ou no formato 
           "D-HH:MM:SS", em caso contrário, sendo  D o número de dias, 
           HH o número de horas, MM o número de minutos e SS o número de
           segundos.
    """

    # Calcula o número de dias, que será igual ao número de segundos 
    # dividido por 86400, que é o número de segundos em um dia. O 
    # operador // é usado para fazer a divisão inteira, que permite 
    # obter o número de dias sem considerar os segundos restantes em um
    # dia não completo. 
    days = time_in_seconds // 86400

    # Calcula o número de horas, que será igual ao número de segundos
    # restantes após a divisão por 86400, dividido por 3600, que é
    # o número de segundos em uma hora. O operador % é usado para obter
    # o número de segundos restantes após a divisão por 86400, e o 
    # operador // é usado para fazer a divisão inteira, que permite 
    # obter o número de horas sem considerar os segundos restantes em 
    # uma hora não completa.
    hours = (time_in_seconds % 86400) // 3600

    # Calcula o número de minutos, que será igual ao número de segundos
    # restantes após a divisão por 3600, dividido por 60, que é o
    # número de segundos em um minuto. O operador % é usado para obter
    # o número de segundos restantes após a divisão por 3600, e o
    # operador // é usado para fazer a divisão inteira, que permite
    # obter o número de minutos sem considerar os segundos restantes em
    # um minuto não completo.
    minutes = ((time_in_seconds % 86400) % 3600) // 60

    # Calcula o número de segundos restantes após a divisão por 60, que
    # é o número de segundos em um minuto. O operador % é usado para
    # obter o número de segundos restantes após a divisão por 60.
    seconds = time_in_seconds % 60

    # Monta a string com o tempo formatado, que inicialmente será 
    # composta por "D-" se o número de dias, armazenado na variável
    # days, for maior que zero, ou uma string vazia em caso contrário, 
    # pois o formato usado pelo SLURM não exige o número de dias se ele
    # for zero.
    prefix = f"{days}-" if days > 0 else ""

    # Retorna a string com o tempo formatado no formato "HH:MM:SS" se o
    # número de segundos for menor que 86400, ou no formato 
    # "D-HH:MM:SS", em caso contrário, utilizando o prefixo de dias
    # armazenado na variável prefix, e formatando os números de horas, 
    # minutos e segundos com dois dígitos, usando o operador de 
    # formatação de strings do Python, que permite preencher com zeros 
    # à esquerda se necessário (o operador :02 indica que o número 
    # deve ter pelo menos dois dígitos, e se tiver menos, será 
    # preenchido com zeros à esquerda).
    return f"{prefix}{hours:02}:{minutes:02}:{seconds:02}"
          
  # Salva em uma string o conteúdo do arquivo de template para o script
  # de submissão da aplicação, definido pelo caminho 
  # template_file_path.
  template_content = template_file_path.read_text(encoding="utf-8")

  # Agora vamos substituir os campos do template pelos parâmetros de
  # configuração do script de submissão definidos no dicionário 
  # template_params, que contém os parâmetros da melhor sugestão de 
  # configuração, o tempo estimado de execução, a partição a ser usada
  # para a submissão do job, o nome do job e todos os parâmetros da 
  # aplicação definidos pelo usuário. 

  # Para alterar o número de nós, primeiramente verificamos se existe a
  # chave 'nodes' no dicionário associado à chave 'suggestion_params' 
  # dentro do dicionário template_params. Se a chave existir, usamos o 
  # valor associado a ela; caso contrário, usamos o valor padrão 1. 
  # Isso é feito usando o método get do dicionário, que retorna o valor 
  # associado à chave se ela existir, ou um valor padrão se não existir,
  # que neste caso será 1.
  number_of_nodes = template_params['suggestion_params'].get('nodes', 1)

  # Altera o campo número de nós no conteúdo do template, substituindo a
  # string "<<number_of_nodes>>" pelo valor de number_of_nodes, que foi
  # obtido como descrito anteriormente, usando o método replace da 
  # classe str.
  template_content = template_content.replace(
    "<<number_of_nodes>>", 
    f"{number_of_nodes}"
  )

  # Para alterar o número de processos por nó, primeiramente verificamos
  # se existe a chave 'process' no dicionário associado à chave 
  # 'suggestion_params'  dentro do dicionário template_params. Se a 
  # chave existir, usamos o valor associado a ela; caso contrário, 
  # usamos o valor padrão 1. Isso novamente é feito usando o método get 
  # do dicionário, usando o valor padrão 1 caso a chave não exista.
  number_of_process = template_params['suggestion_params'].get('process', 1)

  # Altera o campo número de nós no conteúdo do template, substituindo a
  # string "<<number_of_process_per_node>>" pelo valor de 
  # number_of_process, que foi obtido como descrito anteriormente, 
  # usando o método replace da classe str.
  template_content = template_content.replace(
    "<<number_of_process_per_node>>", 
    f"{number_of_process}"
  )

  # Altera o campo número de tarefas , substituindo a string 
  # "<<total_tasks>>" pelo produto do número de nós, armazenado em 
  # number_of_nodes, e o número de processos por nó, armazenado em
  # number_of_process, usando o método replace da classe str.
  template_content = template_content.replace(
    "<<total_tasks>>",
    f"{(number_of_nodes * number_of_process)}"
  )

  # Para alterar o número de threads por processo, primeiramente
  # verificamos se existe a chave 'threads' no dicionário associado à 
  # chave 'suggestion_params'  dentro do dicionário template_params.
  # Se a chave existir, usamos o valor associado a ela; caso contrário, 
  # usamos o valor padrão 1. Isso novamente é feito usando o método get 
  # do dicionário, usando o valor padrão 1 caso a chave não exista.
  number_of_threads = template_params['suggestion_params'].get('threads', 1)

  # Altera o campo número de nós no conteúdo do template, substituindo a
  # string "<<threads_per_process>>" pelo valor de 
  # number_of_threads, que foi obtido como descrito anteriormente, 
  # usando o método replace da classe str.
  template_content = template_content.replace(
    "<<threads_per_process>>", 
    f"{number_of_threads}"
  )

  # Altera o campo nome do job, fornecido pelo usuário na linha de 
  # comando do script de otimização, substituindo a
  # string "<<job_name>>" pelo valor de  da chave 'job_name' do 
  # dicionário template_params, que armazena este nome do job,usando o 
  # método replace da classe str.
  template_content = template_content.replace(
    "<<job_name>>",
    template_params['job_name']
  )

  # Altera o campo com os parâmetros da aplicação, fornecidos pelo
  # usuário na linha de comando do script de otimização, substituindo a
  # string "<<application_params>>" pelo valor de  da chave 
  # 'application_params' do dicionário template_params, que armazena 
  # estes parâmetros da aplicação em uma lista. Como os parâmetros da 
  # aplicação são passados como uma lista de strings, e precisamos 
  # juntar essas strings em uma única string, separadas por espaços, 
  # então usamos o método join da classe str para criar uma string com 
  # todos os parâmetros da aplicação separados por espaços.
  template_content = template_content.replace(
    "<<application_params>>", 
    ' '.join(template_params['application_params'])
  )
  
  # Altera o campo com a partição a ser usada para a submissão do job,
  # obitida a partir do tempo estimado de execução e das sugestões de
  # configuração, substituindo a string "<<partition>>" pelo valor de
  # da chave 'partition' do dicionário template_params, que armazena a
  # partição a ser usada para a submissão do job, usando o método
  # replace da classe str.
  template_content = template_content.replace(
    "<<partition>>",
    f"{template_params['partition']}"
  )

  # Altera o campo com o tempo máximo de execução, obtido a partir do 
  # tempo estimado de execução e das sugestões de configuração. Como o
  # tempo máximo de execução é dado em segundos, precisamos formatá-lo
  # para o formato "D-HH:MM:SS" ou "HH:MM:SS", dependendo do número de 
  # dias, usando a função format_time definida anteriormente. A
  # alteração é feita substituindo a string "<<max_time>>" pelo valor
  # formatado do tempo máximo de execução obtido pela função 
  # format_time, usando o método replace da classe str.
  template_content = template_content.replace(
    "<<max_time>>", 
    f"{format_time(template_params['max_time'])}"
  )

  # Altera o tipo de execução do script de submissão, que pode ser 
  # compartilhada (--oversubscribe) ou exclusiva (--exclusive), 
  # dependendo do valor da chave 'exclusive' do dicionário 
  # template_params, que foi definida de acordo com o esquema da
  # partição escolhida. Se o valor for True, a execução será exclusiva; 
  # caso contrário, será compartilhada. A alteração é feita substituindo
  # a string "<<execution_type>>" pelo valor apropriado, usando o método
  # replace da classe str.
  template_content = template_content.replace(
    "<<execution_type>>", 
    "exclusive" if template_params['exclusive'] else "oversubscribe"
  )

  # Altera o campo uso máximo de memória, obtido a partir do valor da
  # chave 'max_memory' do dicionário template_params, que armazena o
  # valor máximo de memória a ser usado para a submissão do job, também 
  # obtido a partir do esquema da partição escolhida. Como o valor de 
  # memória é dado em kilobytes no arquivo de configuração da aplicação, 
  # precisamos formatá-lo para uma unidade mais legível, como K, M, G, T
  # ou P, usando a função format_size definida anteriormente, 
  # multiplicando o valor por 1024 para convertê-lo para bytes antes de
  # chamar a função format_size, pois a função espera que o tamanho 
  # esteja em bytes. A alteração é feita substituindo a string 
  # "<<max_memory>>" pelo valorformatado obtido pela função format_size, 
  # usando o método replace da classe str. 
  template_content = template_content.replace(
    "<<max_memory>>", 
    f"{format_size(template_params['max_memory'] * 1024)}"
  )

  # Retorna o conteúdo do script de submissão gerado a partir do arquivo
  # de template, como descrito anteriormente, em uma string do Python.
  return template_content

# Script de monitoração: Iria, de tempos em tempos, para cada aplicação,
# pegar cada coluna JobID e verificar se o job já terminou e, em caso,
# positivo, adicionar as informações relevantes obtidas pelo sacct aos
# dados do job.

def submission_log(application_config, system_config, suggestion, 
                   template_params, required_applicaion_params,  
                   user_application_params, application_args, job_id, 
                   verbose=False):
  """
  Função para gerar o log de submissão do trabalho para a aplicação, 
  quando esse trabalho for automaticamente submetido pelo usuário, pois
  os dados do trabalho submetido, como por exemplo o tempo de execução o
  consumo de energia, e o EDP, serão usados como novos dados de
  execuções reais para treinamentos futuros dos modelos. Para cada 
  aplicação que pode ser otimizada pelo script de otimização e para a
  qual um usuário usou o script para otimizá-la será salvo, no log desta
  aplicação, um arquivo no formato .csv com as linhas com cada execução
  da aplicação otimizada e as colunas as informações referentes às
  variáveis relaciondas à execução da aplicação. As informações salvas
  serão às relacionadas a prórpia submissão do trabalho, como no nome do
  trabalho, a partição usada, o tempo máximo de execução nesta partição,
  o uso máximo de memória, se o trabalho deve ou não ser executado de
  modo exclusivo, as variáveis relacionadas à melhor sugestão de
  configuração (por exemplo, número de nós, processos por nó e threads
  por processo) que terão os mesmos nomes usados ao treinar o modelo da
  aplicação, às variáveis da aplicação definidas pelo usuário ao oimizar
  a execução da aplicação, as variáveis que foram obtidas pelas
  convwersões das variáveis de aplicação que foram fornecidas pelo
  usuário e que são efetivamente usadas ao treinar o modelo da
  aplicação, o tempo de execução estimado ao executar a alicação usando
  os parâmetros da aplicação convertidos e a melhor sugestão de
  configuração, e o valor mínimo da variável alvo (no nossos estudos, a
  EDP) usada pelo modelo auxiliar ao escolher a melhor sugestão de
  configuração, o identificador do trabalho, que será usado
  posteriormente para obter as informações referentes à execução deste
  trabalho e necessárias para treinar o modelo considerando as 
  informações obtidas pela execução desse trabalho, e todos os 
  parâmetros da aplicação passados pelo usuário, incluindo os usados
  para obter os parâmetros convertidos e efetivamente usados ao treinar
  os modelos.

  Parâmetros:
    application_config (dict): Dicionário com as configurações gerais da
                               aplicação que está sendo otimizada pelo
                               script. Vamos usar a chave 'name' nas 
                               possíveis mensagens de erro e o
                               dicionário referenciado pela chave 'user'
                               para obter as informações sobre as 
                               varáveis alvo do modelo, com o objetivo
                               de criar colunas com nomes que facilitem
                               a identificação dessas variáveis alvo.
    system_config (dict): Dicionário com as configurações gerais
                          relacionadas aos scripts de otrimização e de
                          treinamento. No caso da função, iremos usar
                          chave 'logs_path' com o nome do diretório em
                          que os logs são armazenados.  
    suggestion (dict): Dicionário com as informações referentes à melhor
                       sugestão de confiuração. Vamos usar a chave 
                       'suggestion' com os valores das variáveis de
                       configuração da melhor sugestão de configuração,
                       a chave 'time' com o tempo de execução estimado
                       para a melhor sugestão, e a chave 
                       'y_pred_mininum' com o melhor valor da variável
                       alvo usado ao escolher a melhor sugestão de
                       configuração, que foi o valor predito
                       considerando essa sugestão. 
    template_params (dict): Dados referentes ao script de submissão não
                            relacionados as variáveis de configuração,
                            ou seja, o tempo máximo de execução, o uso
                            máximo de memória, se o trabalho é ou não
                            exclusivo, e a partição usada.
    required_applicaion_params (NameSpace): Objeto do tipo NameSpace
                                            retorado pela função parser
                                            do objeto da classe 
                                            argparse.ArgumentParser com
                                            os parâmetros da aplicação
                                            que o usuário precisa 
                                            necessariamente fornecer
                                            para ser possível otimizar a
                                            aplicação.
    user_application_params (dict): Dicionário com os parâmetros da
                                    aplicação obtidos por conversões a
                                    partir dos parâmetros obrigatórios
                                    da aplicação fornecedos pelo usuário
                                    e passados no objeto
                                    required_applicaion_params.  
    application_args list[str]: Lista com todos os parâmetros da 
                                aplicação passados pelo usuário,
                                incluindo os usados e convertidos ao
                                escolher a melhor sugestão de
                                configuração.
    job_id (int): Identificador do trabalho submetido ao SLURM pelo 
                  programa de submissão, por exemplo, o sbatch.


    Retorna:
      bool: True se o arquivo de log foi atualizado com sucesso, ou 
            False se algum erro ocorreu ao criar (o primeiro log da
            aplicação) ou acessar o arquivo de log.    
  """

  # Armazena uma referência para o nome da aplicação atualmente sendo
  # otimizada pelo usuário na variável application_name.
  application_name = application_config['name']

  # Armazena uma referência para os parâmetros usados ao treinar os
  # modelos, com o objetivo de definir o correto nome das colunas com
  # o tempo de execução predito para a melhor sugestão de configuração e
  # o valor mínimo da variável alvo predita pelo modelo auxiliar usado
  # para escolher a melhor sugestão (lembrando que, no nosso estudo, a
  # variável do modelo auxiliar é a EDP e a do modelo de predição do
  # tempo é a ElapsedRaw).
  estimated_parameters = application_config["estimated_parameters"]

  # Primeiramente copiamos todos os campos do dicionário 
  # template_params, usados para gerar o script de submossão,
  # com exceção do campo 'application_params' e do campo 
  # 'suggestion_params', pois precisamos usar os parâmetros da
  # aplicação e da sugestão de configuração, pois esses são os
  # nomes usados ao treinar os modelos. Também removemos o
  # campo 'application_name', pois existe um arquivo de log
  # diferente para cada aplicação. 
  logs_params = {
      k: v
      for k, v in template_params.items()
      if k not in [
        'application_params', 'suggestion_params', 'application_name'
      ]
  }            

  # Adiciona ao dicionário logs_params os parâmetros da 
  # melhor sugestão de configuração.
  logs_params.update(suggestion['Suggestion'])

  # Adiciona ao dicionário logs_params os parâmetros da
  # aplicação definidos pelo usuário.
  logs_params.update(vars(required_applicaion_params))

  # Agora adiconamos os parâmetros da aplicação convertidos,
  # que são os parâmetros usados pelos treinamentos ao
  # construir o preditor de aplicação e por este preditor ao
  # escolher a melhorsugestão de configuração.
  logs_params.update(user_application_params)

  # Adicioma ao dicionário logs_params o tempo predito para a
  # sugestão de configuração.
  if 'Time' in suggestion.keys():
    logs_params[
      f'predicted {estimated_parameters["time"]}'
    ] = suggestion['Time']

  # Adicioma ao dicionário logs_params o menor valor predito
  # para variável alvo usada pelo modelo auxiliar ao fazer as
  # predições (nos testes atuais, estamos usando o EDP).
  logs_params[
    f'predicted {estimated_parameters["suggestion"]}'
  ] = suggestion['y_pred_minimum']

  # Por fim, adicionamos o ID do job submetido ao dicionário
  # logs_params.
  logs_params['JobID'] = job_id

  # Adicionamos uma coluna com todos os parâmetros da aplicação passados
  # pelo usuário.
  logs_params['Application Params'] = ', '.join(application_args)

  # Define as colunas do DataFrame que será usado para armazenar os logs
  # das submissões feitas para a aplicação, que serão salvos em um
  # arquivo .csv. As colunas serão as chaves do dicionário log_params,
  # que contém os parâmetros da sugestão de configuração, o tempo
  # estimado de execução, a partição a ser usada para a submissão do
  # job, o nome do job e todos os parâmetros da aplicação definidos
  # pelo usuário, os parametros da aplicação após as conversões feitas
  # pelo script de otimização, o menor valor predito para a variável
  # alvo de otimização, o menor valor predito para a variável alvo
  # usada pelo modelo auxiliar para escolher a melhor sugestão de 
  # configuração.
  logs_columns = list(logs_params.keys())

  # Define o nome do arquivo em que serão armazenados os logs das
  # submissões feitas para a aplicação cujas configurações estão em
  # application_args.
  try:
    log_filepath = (base_files_path / Path(system_config['logs_path']) 
                    / f"{application_config['user']['log_file']}")
    if log_filepath.is_file():
      # O arquivo com os logs de otimização dos usuários para a
      # aplicação a ser otimizada existe, e acessível e é um arquivo.
      # Logo, vamos ler o arquivo para um Dataframe para depois
      # atualizá-lo com os dados do trabalho submetido para a aplicação
      # otimizada.
      logs_df = pd.read_csv(log_filepath, sep='|', usecols=logs_columns)

      # Como já existe um dataframe com os logs, vamos atualizá-lo com
      # os dados do trabalho submetido para a aplicação otimizada, que
      # estão em log_params.
      logs_df = pd.concat([logs_df, pd.DataFrame(logs_params, 
                                                 index=[0])])
      # Recria os índices do dataframe com os logs.
      logs_df = logs_df.reset_index(drop=True)

    elif log_filepath.is_dir():
      # Se o caminhio for um diretório, 
      print("❌ Erro ao ler o arquivo com os logs para a aplicaçao "
            f"{application_name}! O caminho {log_filepath.resolve()} é um "
            "diretório!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

      # Como o caminho do arquivo de log aponta na verdade para um
      # diretório, o que não deveria ocorrer, então retornamos False.
      return False

    elif log_filepath.exists():
      # Se o caminhio nao for um arquivo e nem um diretório, 
      print("❌ Erro ao ler o arquivo com os logs para a aplicaçao "
            f"{application_name}! O caminho {log_filepath.resolve()} é um "
            "arquivo especial do sistema operacional!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

      # Como o caminho do arquivo de log aponta na verdade para um
      # objeto que não é um arquivo e nem um diretório, o que não 
      # deveria ocorrer, então retornamos False.
      return False

    else:
      # Este caso é quando o caminho é inválido, vamos partir da suposição
      # de que o arquivo não existe, significando que este pode ser o
      # primeiro trabalho submetido para a aplicação depois dela ser
      # disponibiliada para ser otimizada pelo script. Logo, deveremos
      # criar um dataframe vazio com as colunas definidas em
      # logs_columns.
      logs_df = pd.DataFrame(logs_params, index=[0])
    
    # Se a verbosidade estiver habilitada, imprime o dataframe 
    # atualizado com os logs da aplicação para a otimização que foi 
    # feita (a última linha da tabela), e as outras otimizações 
    # anteriores, se existirem. Esta variável deve ser setada, na função
    # de otimização, pela variável de ambiente para depuração do código,
    # pois não tem sentido o usuário ver este dataframe.
    if verbose:
      print("⚠️ Dataframe com os logs das otimizações feitas pelo script de "
            f"otimização para a aplicação {application_name}:")
      print("\n", logs_df.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", 
            sep="")
   
    # Salva o DataFrame atualizado com os logs das submissões feitas
    # para a aplicação otimizada em um arquivo .csv, que será criado se
    # ainda não existir, ou sobrescrito se já existir, usando o
    # método to_csv.
    logs_df.to_csv(log_filepath, sep='|', index=False, mode='w')

    # Como conseguimos atualizar o arquivo de log da aplicação com
    # sucesso, então retornamos True.
    return True
  
  except FileNotFoundError as e:
    print(f"❌ O arquivo {e.filename} nao foi encontrado.")
    print("❌ Por favor, avise o erro ao adistrador do sistema o erro: "
          f"{e.strerror}!")

    # Como oocorreu una exceção inesperada, retorna None ao invés do 
    # dicionário com os valores convertidos das variáveis de aplicação.
    return None

  # Não foi possível acessar um dos arquivos, devido a um erro de
  # permissão de acesso ao arquivo.
  except PermissionError as e:
    print(f"❌ Erro de permissão ao acessar o arquivo {e.filename}.")
    print("❌ Por favor, avise o erro ao adistrador do sistema o erro: "
          f"{e.error}.")
    # Como ocorreu uma exceção, retorna false para indicar que ocorreu
    # um erro ao tentar gerar o log de submissão do trabalho para a
    # aplicação do usuário otimizada pelo script de otimização..
    return False

  # Não foi possível acessar um dos arquivos, devido a um erro de
  # leitura ao acessar o arquivo.
  except IOError as e:
    print(f"❌ Erro de I/O ao ler o arquivo {e.filename}!")
    print(f"❌ Código do erro: {e.errno}; Mensagem: {e.strerror}!")
    print(f"❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como ocorreu uma exceção, retorna false para indicar que ocorreu
    # um erro ao tentar gerar o log de submissão do trabalho para a
    # aplicação do usuário otimizada pelo script de otimização..
    return False

  # Devido a um erro no código, foi tentado o acesso de um campo de um
  # objeto referenciado por uma chave inexistente.
  except KeyError as e:
    if application_name is None:
      print("❌ Erro ao processar uma das estruturas indexadas por chave, a "
            f"chave {e.args[0]} não existe, ao inicializar as otimizações!")
    else:
      print("❌ Erro ao processar uma das estruturas indexadas por chave, a "
            f"chave {e.args[0]} não existe, ao otimizar a aplicação "
            f"{application_name}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como ocorreu uma exceção, retorna false para indicar que ocorreu
    # um erro ao tentar gerar o log de submissão do trabalho para a
    # aplicação do usuário otimizada pelo script de otimização..
    return False

  # Quando tentamos acessar um objeto, um erro no código tentou acessar
  # um atributo que não existe no objeto.
  except AttributeError as e:
    print("❌ Erro ao acessar um dos objetos internos do script, um dos "
          "seus atributo não existe, ao otimizar a aplicação "
          f"{application_name}!")
    print(f"❌ Erro gerado: '{e.args[0]}'")  
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como ocorreu uma exceção, retorna false para indicar que ocorreu
    # um erro ao tentar gerar o log de submissão do trabalho para a
    # aplicação do usuário otimizada pelo script de otimização..
    return False

  # Quando tentamos usar algum valor, um valor incorreto nçao suportado
  # ou fora de uma faixa de valore, foi usado.
  except ValueError as e:
    print("❌ Erro ao inicializar ou usar estruturas internas do script, ao "
          f"fazer a otimização para a aplicação {application_name}!")
    print(f"❌ Erro gerado: '{e.args[0]}'")  
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como ocorreu uma exceção, retorna false para indicar que ocorreu
    # um erro ao tentar gerar o log de submissão do trabalho para a
    # aplicação do usuário otimizada pelo script de otimização..
    return False

  # Ocorreu alguma outra exceção inesperada.
  except Exception as e:
    print("❌ Erro desconhecido ao fazer a otimização da aplicação "
          f"{application_name}!")
    print(f"❌ Parâmetros do erro: {e.args}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como ocorreu uma exceção, retorna false para indicar que ocorreu
    # um erro ao tentar gerar o log de submissão do trabalho para a
    # aplicação do usuário otimizada pelo script de otimização..
    return False

def find_application_config(applications_config, application_name):
  """
  Função para procurar e retornar a chave referenciando as configurações
  de uma aplicação identificada pelo seu executável, que pode ser um dos
  nomes de executáveis dados na lista referenciada pela chave 
  'executable_names' do dicionário refenciado pela chave 'user' do 
  dicionário com as configurações de uma aplicação. Será escolhida a
  primeira aplicação para a qual o nome do executável está na sua lista
  de executáveis logo, para evitar inconsistências, duas aplicações 
  diferentes, mesmo que sejam variações da mesma aplicação, por exemplo,
  compiladas para usar somente MPI ou MPI com OpenMP, não podem ter o 
  mesmo nome para o executável.

  applications_config (dict): Dicionário com todas as configurações das
                              aplicações que podem ser otimizadas pelo
                              script de otimização, sendo que cada
                              chave, associada e identificando uma
                              aplicação, tem uma referência para o
                              dicionário com as configurações dessa
                              aplicação.
  application_name (str): Nome de um dos possíveis arquivos executáveis
                          da aplicação, passados pelo usuário como o 
                          primerio parâmetro da aplicação, que
                          identifica a aplicação que o usuário está
                          tentando otimizar (o parâmetro que vem logo
                          após do separador '--').

  Retorna:
    str | None: String com o nome que identifica a aplicação, ou seja, a
                chave no dicionário applications_config com a referência
                para o dicionário com as configurações da aplicação que 
                possui como possíveis nomes de executáveis o nome dado
                em application_name. Se application_name não for achado
                em nenhum dos nomes de executáveis, considerando cada
                aplicação definida pelas chaves do dicionário
                applications_config, então será retornado None para
                indicar que esta aplicação não é ainda suportada e
                portanto não pode ser otimizada.
  """

  # Iniializa a variável que armazenará a chave da aplucação no
  # dicionário applications_config com None, inicialmente indicando que
  # a aplicação cujo executável é passado em application_name não é
  # suportada.
  application_id = None

  # Para verificar se application_name é um possível nome de executável
  # de uma das aplcações suportadas, precisamos percorrer todas as
  # chaves do dicionário applications_config.
  for application_id_aux in applications_config.keys():
    # Verifica se application_name é um dos executáveis da aplicação
    # para a qual estamos verificando os seus possíveis executáveis,
    # identificada pela chave application_id_aux. Para fazer isso, basta
    # verificar se a string application_name está dentro (é uma das
    # strings, usando o operador in) da lista definida pela chave
    # 'executable_names' do dicionário referenciado pela chave 'user',
    # comctodos os possíveis nomes dos executáveis para a aplicação
    # identificada por application_id_aux.
    if application_name in (
      applications_config[application_id_aux]['user']['executable_names']
    ):
      # Se o application_name for um dos nomes dos executáveis, então
      # encontramos a chave no dicionário applications_config, ou seja,
      # a string application_id_aux. Logo, definimos application_id, com
      # a chave da aplicação para a referência application_id_aux, com o
      # nome da chave da aplicação, e encerramos a verificação, pois já
      # encontramos a chave identificando a aplucação a ser otimizada.
      application_id = application_id_aux
      break

  # Retorna a variável application_id, que terá a chave do dicionário
  # applications_config com a refência da configuração da aplicação a
  # ter a sua execução otimizada se application_name for um dos
  # possíveis executáveis de uma das aplicações ou None em caso
  # contrário, para indicar que a aplucação não pode ter a sua execução
  # otimizada.
  return application_id

def list_applications(applications_config, script_config):
  for application_id in sorted(applications_config.keys()):
    print(f"➡️  Aplicação {application_id}, possíveis nomes para os "
          f"executáveis: {', '.join(script_config['executable_names'])}")
  return True

def get_custom_configurations(application_name, application_id, 
                              script_config, user_args, 
                              application_partitions_list):
  # Se o usuário definir pelo menos uma opção, usa a configuração 
  # customizada com as outras opções com valores defaault se não 
  # definidas pelo usuário.    

  # Verifica se o usuário usou as opções número de nós, de processos por nó, e de threads por processo.
  # Primeiramemte verifica se o usuário definiu alguma das opções de configuração;
  use_custom_config = False
  for suggestion_name in script_config['suggestions_map']:
    if not hasattr(user_args, suggestion_name):
      print(f"❌ A configuração necessária {suggestion_name} não existe nas "
            f"opções do script para a aplicação {application_id}.")
      print(f"❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return False
    elif getattr(user_args, suggestion_name) is not None:
      use_custom_config = True

  if use_custom_config:
    custom_configurations = {}
    for suggestion_name in script_config['suggestions_map']:
      suggestion_value = getattr(user_args, suggestion_name)
      custom_params = get_options_suggestion(suggestion_value)
      if custom_params is None:
        print(f"❌ Erro de sintaxe ao processar a opção --{suggestion_name} "
              f"com o valor {suggestion_value}!")
        return False
      max_value_custom_params = max(custom_params)
      max_possible_value = max(
        [partition[suggestion_name] for partition in application_partitions_list]
      )
      if max_value_custom_params > max_possible_value:
        application_partitions_names = {partition['partition'] for partition in application_partitions_list}
        print("⚠️  Descartando todos os valores para a opção \033[31m"
              f"{suggestion_name}\033[0m maiores do que {max_possible_value} "
              "permitidos pelas possíveis partições \033[1;34m"
              f"{', '.join(application_partitions_names)}\033[0m da aplicação "
              f"{application_name}!")
        
        custom_params = [
          custom_value for custom_value in custom_params 
            if custom_value <= max_possible_value
        ]                 

      custom_configurations[
        script_config['suggestions_map'][suggestion_name]
      ] = custom_params
  else:
    custom_configurations = None

  return custom_configurations

def process_application_params(application_name, script_config, 
                               application_configs_dir_path):
  
  # Processa os parâmetros da aplicação.
  parser_application = argparse.ArgumentParser(
    description="Esta ajuda descreve os parâmetros da aplicação que "
                "precisam ser obrigatoriamente definidos.", 
    prog=application_name, 
    add_help=False, 
    formatter_class=CustomFormatter
  )
  application_params = script_config['user_options']
  opcoes = parser_application.add_argument_group("Opções principais")
  ajuda = parser_application.add_argument_group("Ajuda")
  ajuda.add_argument(
    "-h", "--help", 
    action="help", 
    help="Mostra esta mensagem de ajuda e sai."
  )
  for param in application_params.keys():
    opcoes.add_argument(
      *application_params[param]['params'], 
      required=True, 
      help=application_params[param]['help'], 
      type=get_type(application_params[param]['type']), 
      dest=param
    )

  # Converte os argumentos da aplicação para o dicionário a ser usado
  # pela função de predição.                                  
  (
    required_applicaion_params, other_applicatios_params
  ) = parser_application.parse_known_args(application_args[1:])

  # Processa os patâmetros usados pela aplicação para o preditor. 
  user_application_params = convert_user_params(required_applicaion_params, 
                                                script_config['conversions'], 
                                                application_configs_dir_path)  
  
  return (required_applicaion_params, other_applicatios_params, 
          user_application_params)

def get_template_params(application_name, user_args, application_args, 
                        application_partitions_list, suggestion, 
                        reversed_suggestions_map):

  suggestion_mapped = {
    reversed_suggestions_map[k]:v 
    for k,v in suggestion['Suggestion'].items()
  }

  # Cria o dicionário com as informações para construir o script de 
  # submissão (fiz o dicionário para tornar a função independente de como os parâmetros são gerados).
  #list_partitions = script_config['slurm']
  template_params = {
    'application_name': application_name,
    'suggestion_params': suggestion_mapped,
    'job_name':  application_name if user_args.jobname is None else user_args.jobname,
    'application_params': application_args[1:],
  }

  # Verifica se existe o tempo predito para a sugestão.
  if 'Time' in suggestion.keys():
    # Aproxima o tempo para o maior tempo inteiro.
    predicted_time = np.ceil(suggestion['Time'])
  else:  
    predicted_time = 0
  # Verifica quais partições podem ser usadas pela sugestão.
  valid_partitions_list = []
  for partition in application_partitions_list:
    # Descobre quais partições podem executar a aplicação.
    valid_partition = partition['max_time'] >= predicted_time
    for suggestion_name in suggestion['Suggestion']:
      partition_suggestion_name = reversed_suggestions_map[suggestion_name]
      valid_partition = valid_partition and partition[partition_suggestion_name] >= suggestion['Suggestion'][suggestion_name]
    if valid_partition:
      valid_partitions_list.append(partition)

  # Se existirem partições, escolhe a com o menor tempo (portanto, mas próximo do tempo da aplicação, já que todas as partiçoes da lista)
  if valid_partitions_list:                                           
    # A partição escolhida será a com menor tempo máximo.
    partition_used = min(valid_partitions_list, key=lambda partition: partition['max_time'] - predicted_time)
  else:   
    # A partição usada será a default (para evitar erros, a partição default deveria ser a com todos os recursos que sugerimos com 
    # os valores. A list compreension deveria retornar somente um gerador com somente um elemento, obtido com o next, 
    # pois o validados do JSON deveria impedir mais de uma partição com o dafault igual a true e também todas as partições com 
    # o default igual a false.
    partition_used = next([partition for partition in application_partitions_list if partition["dafault"]])
    # Verifica se o tempo da partição é maior do que o temṕo predito, e sá um aviso se isso ocorrer
    if predicted_time > partition_used['max_time']:
      print(f"⚠️  O tempo predito aproximado {predicted_time} é maior do que o tempo máximo {partition_used['max_time']} de execução da partição {partition_used['partition']}!")

  # define os dados para gerar o script de confuguração.
  template_params['partition'] = partition_used['partition']
  template_params['max_time'] = partition_used['max_time']
  template_params['max_memory'] = partition_used['max_memory']  
  template_params['exclusive'] = partition_used['exclusive']  

  return template_params


def submit_submission_script(script_file_name, template_content):
  try:
    remove_temp_file = False
    job_id = None
    if script_file_name is None:
        remove_temp_file = True
        with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as temp:
            temp.write(template_content)
            script_file_name = temp.name 

        # TODO: Depois podemos remover, se necessário, este código de 
        # depuração.
        # TODO: Início.
        if debug_code:
          print(f"➡️  Arquivo temporário {script_file_name} criado para "
                "armazenar o script de submissão.")
        # TODO: Fim  
  
    # Executa o sbatch se a opção -r ou --run foi usada
    submission_program = user_config["slurm"]["submission_program"]
    result = subprocess.run([submission_program, f"{script_file_name}"], capture_output=True, text=True, check=True)   

    print("➡️  Script de submissão submetido com sucesso!")

    # Imprime para o usuário o ID do job submetido.
    # Extrai o ID da saída do sbatch
    job_id_regex = re.compile(user_config['slurm']['submission_message']) 
    resultado = re.search(job_id_regex, result.stdout)
    job_id = resultado.group(1) 
    print(f"➡️  O trabalho foi submetido com o identificador {job_id}.")

    # TODO: Depois podemos remover, se necessário, este código de depuração.
    # TODO: Início.
    if debug_code:
      print(f"➡️  O código de retorno da execução do programa de submissão {submission_program} foi {result}")
      print("➡️  O campo returncode do objeto CompletedProcess deveria ser 0, pois um valor diferente de 0 deveria gerar a exceção subprocess.CalledProcessError.")
    # TODO: Fim  

    # Imprime a saída da execução do programa de submissão do script.
    # TODO: Está correto isso ser um vernose? Talvez usar a variáel global debug_code?        
    if user_args.verbose:
      print(f"➡️  stdout da execução de {submission_program}:\n\n")
      print(result.stdout)
      print(f"\n\n➡️  stderr de execução de {submission_program}:\n\n")
      print(result.stderr)        

  except subprocess.CalledProcessError as e:
    # This will print the actual error from the terminal command
    print("❌ Não foi possṕivel executar o comando {submission_program}!")
    print("❌ Código de saída:", e.returncode)
    print("❌ mensagem de erro:", e.stderr)
  except FileNotFoundError:
    print(f"❌ O programa {submission_program} não foi achado no sistema!")
    print(f"❌ Por favor, reporte este erro ao adminstrador do sistema!")
  except IOError as e:
    print(f"❌ Não foi possível criar o arquivo temporário. {e.filename}")
    print(f"❌ Código do erro: {e.errno}; Mensagem: {e.strerror}!")
    print(f"❌ Por favor, reporte este erro ao adminstrador do sistema!")
  finally:
    # Se criamos um arquivo temporario, removemos depois de usarmos.
    if remove_temp_file:
      # TODO: Depois podemos remover, se necessário, este código de depuração.
      # TODO: Início.
      if debug_code:
        print(f"➡️  Tentando remover o arquivo temporário {script_file_name}.")
      # TODO: Fim  
      try:
        if os.path.exists(script_file_name):
          os.remove(script_file_name)    
        if debug_code:
          print(f"➡️  Arquivo temporário {script_file_name} removido com sucesso.")
      except OSError as e:
        if debug_code:
          print(f"⚠️  Não foi possível remover o arquivo {e.filename}")
          print(f"⚠️  Código do erro: {e.errno}; Mensagem: {e.strerror}!")
        Status = False
  return job_id

def optimize_application(configs_file_path, system_config, applications_config, 
                         user_config, application_args, predictors_info_config, 
                         user_args):
  """
  Função principal do script de otimização, que recebe os argumentos do
  sistema, os argumentos do usuário, os parâmetros da sugestão de configuração 
  e todos os parâmetros da aplicação definidos pelo usuário, e que será usada 
  para otimizar a execução da aplicação, gerando o script de submissão para a
  aplicação, que será salvo em um arquivo de log. A função retorna True se a
  otimização foi bem-sucedida, ou False caso contrário.
  """

  # Verifica se o usuário deseja somente listar as aplicações
  application_name = None
  try:
    if user_args.list:
      return list_applications(applications_config, script_config)
    else:
      # Caso não deseje listar as aplucações, precisamos fornecer uma 
      # aplicaçao, pois o usuário deseja otimizar o uso dos reursos.
      if not application_args:
        print("❌ Não foi fornecido o nome da aplicação a ser otimizada e os "
              "seus parâmetros de execução.")
        return False
      
      # Diretorio dos arquivos de configuração das aplicações.
      application_configs_dir_path = configs_file_path / 'applications'

      # O nome da aplicação é o primeiro parâmetro da lista de
      # parâmetros da apliação passados pelo usuário.
      application_name = application_args[0]
 
      # Verifica, pelo nome da aplicação, dado como o primeiro rgumento
      # dos parâmetros da aplicação passados pelo usuário, se existe
      # uma configuração para esta applicação. Se existir, retorna a 
      # ID (chave) desta applicação no dicionário applications_config.
      # Em caso contrário, retorna None
      application_id = find_application_config(applications_config,
                                               application_name)
      if application_id is None:
        print(f"⚠️  A otimização para a aplicação {application_name} ainda "
              "não é suportada!")  
        return False

      # Para facilitar o acesso, define uma variável com uma referência
      # para o dicionário con as configurações da apliação definida pela
      # chave application_id.
      application_config = applications_config[application_id]

      # Para facilitar o acesso, define uma variável com uma referência
      # para as configurações da aplicação usadas pelo script de
      # otimização;
      script_config = application_config['user']

      application_partitions_list = script_config['slurm']

      (
        required_applicaion_params, other_applicatios_params, 
        user_application_params
      ) = process_application_params(application_name, script_config, 
                                     application_configs_dir_path)

      if user_application_params is None:
        return False

      custom_suggestions = get_custom_configurations(
        application_name, 
        application_id, 
        script_config, user_args,
        application_partitions_list
      )

      # Lê o preditor usado para fazer a melhor sugestão dos parâmetros
      # de execução da aplicação.
      predictor_path = (base_files_path 
                        / Path(system_config['predictors_path']) 
                        / predictors_info_config[application_id]
      )
      predictor = SuggestionsPredictor.load_predictor(predictor_path)

      suggestion = predictor.get_suggestion(user_application_params, 
                                            custom_suggestions, 
                                            verbose=debug_code)

      # Cria o mapeamento reverso para a impressao
      suggestions_map = script_config['suggestions_map']
      reversed_suggestions_map = {
        v:k for k, v in suggestions_map.items()
      }

      # Obtém o caminho do arquivo de template, se as opçoes. 
      if user_args.run or not user_args.suggestion:
        if user_args.verbose:
          SuggestionsPredictor.print_suggestion(
            suggestion, 
            suggestion_map=reversed_suggestions_map, 
            show_time=True,
            show_score=True, 
            show_X=True, 
            show_y_pred=True
          )

        template_params = get_template_params(application_name, user_args, 
                                              application_args, 
                                              application_partitions_list,
                                              suggestion, 
                                              reversed_suggestions_map)

        template_file_path = (base_files_path 
                              / Path(system_config['templates_path']) 
                              / script_config['script_template_name']
        )
        template_content = generate_submission_script(template_file_path, 
                                                      template_params)

        if user_args.verbose:      
          print("➡️  Script de submissão: \n")
          print(template_content)
          print()

        # Salva no arquivo passado como parâmetro ou o nome default definido no arquivo de cofiguração do usuário
        # Se o usuário não fornecer um nome pela opção -s ou --script, cria um arquivo temporário.
        if user_args.script or not user_args.suggestion:
          # Salva o script de su
          try:
            if user_args.script is None:
              script_file_name = user_config["default_script_name"]
            else:     
              script_file_name = user_args.script
            with open(script_file_name, "w", encoding="utf-8") as script_file:
              script_file.write(template_content)
            print(f"➡️  Script de submissão {script_file_name} criado com sucesso!")
          except PermissionError as e:
            print(f"❌ Erro de permissão ao acessar o arquivo {script_file_name}!")
            return False
          except IOError as e:
            print(f"❌ Erro de I/O ao ler o arquivo {script_file_name}!")
            print(f"❌ Código do erro: {e.errno}; Mensagem: {e.strerror}!")
            return False
        else:
          script_file_name = None

        if user_args.run:
          job_id = submit_submission_script(script_file_name, 
                                            template_content)
          if job_id is not None and user_config['enable_submission_log']:
            # Gera o log de submissão do trabalho da aplicação
            # otimizada, que será usado para monitorar o trabalho
            # submetido, e que será salvo em um arquivo de no formato
            # csv, com todas as informações referentes à otimização do
            # trabalho (melhor sugestão de configuração, parâmetros da
            # aplicação passados pelo usuário e parâmetros da
            # aplicação convertidos, nome do trabalho, tempo de
            # execução estimado para a melhor configuração, menor 
            # da variável alvo do modelo auxiliar usado para descobrir
            # a melhor sugestão e ID do trabalho ao ser submetido pelo
            # sbatch).

            submission_log(application_config, 
                            system_config, suggestion, template_params,
                            required_applicaion_params, 
                            user_application_params,
                            application_args[1:], job_id, debug_code)
      else:
        SuggestionsPredictor.print_suggestion(
          suggestion, 
          suggestion_map=reversed_suggestions_map, 
          show_score=user_args.verbose, 
          show_X=user_args.verbose, 
          show_y_pred=user_args.verbose, 
          show_time=user_args.verbose
        )

      return True
  except KeyError as e:
    if application_name is None:
      print("❌ Erro ao processar uma das estruturas indexadas por chave, a "
            f"chave {e.args[0]} não existe, ao inicializar as otimizações!")
    else:
      print("❌ Erro ao processar uma das estruturas indexadas por chave, a "
            f"chave {e.args[0]} não existe, ao otimizar a aplicação "
            f"{application_name}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
    return False
  except AttributeError as e:
    if application_name is None:
      print("❌ Erro ao acessar um dos objetos internos do script, um dos "
            "seus atributo não existe, ao inicializar as otimizações!")
    else:
      print("❌ Erro ao acessar um dos objetos internos do script, um dos "
            "seus atributo não existe, ao otimizar a aplicação "
            f"{application_name}!")
    print(f"❌ Erro gerado: '{e.args[0]}'")  
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
    return False
  except ValueError as e:
    if application_name is None:
      print("❌ Erro ao inicializar ou usar estruturas internas do script, ao "
            "inicializar as otimizações!")
    else:
      print("❌ Erro ao inicializar ou usar estruturas internas do script, ao "
            f"fazer a otimização para a aplicação {application_name}!")
    print(f"❌ Erro gerado: '{e.args[0]}'")  
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
    return False
  except Exception as e:
    if application_name is None:
      print("❌ Erro desconhecido ao começar a fazer as otimizações!")
    else:  
      print("❌ Erro desconhecido ao fazer a otimização da aplicação "
            f"{application_name}!")
    print(f"❌ Parâmetros do erro: {e.args}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

# Processa a linha de comando.
user_args, application_args = process_script_args()

# Verifica se algum erro ocorreu
if user_args is None:
  exit(-1)

# Lê os arquivos de confoguraçã.o
configs_file_path, system_config, applications_configs, user_config, predictors_info_config = read_configs(verbose=user_args.verbose)

if system_config is None or applications_configs is None or user_config is None or predictors_info_config is None:
  print('❌ Erro ao processar um dos arquivos de configuração. Por favor, avise o erro ao suporte.')
  exit(-3)

# Processa os parâmetros da linha de comando
status = optimize_application(configs_file_path, system_config, applications_configs, user_config, application_args, 
                              predictors_info_config, user_args)
if not status:
  print("Erro ao otimizar o script!")
  exit(-2)
