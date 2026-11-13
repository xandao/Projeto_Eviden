# Bibliotecas do Python usadas pelo código do script de treinamento.
import pandas as pd
from Utils.Suggestions import (
  FilterOutliers, 
  BestHiperparams, 
  DiscoverBestModel, 
  SuggestionsPredictor
)
from Utils.ReadConfigs import (
  ReadSystemConfig, 
  ReadApplicationsConfigs, 
  ReadTrainingConfig, 
  PredictorsInfoConfig
)
import importlib
import argparse
import sys
import textwrap
from pathlib import Path
from functools import partial
from Utils.Common import (
  BASE_FILES_PATH_ENV_NAME, 
  BASE_FILES_PATH, 
  CONFIGS_FILES_DIR, 
  DEBUG_CODE, 
  CustomFormatter
)

def read_configs(verbose=False):
  """
  Lê os arquivos de configuração do sistema, das aplicações e do
  usuário. Retorna uma tupla com os seguintes elementos, referentes às
  configurações lidas:

  - configs_file_path: Caminho para o diretório de configuração.
  - system_config: Dicionário com as configurações do sistema.
  - applications_configs: Dicionário com as configurações das 
    aplicações.
  - training_config: Dicionário com as configurações de treinamento dos
                     modelos a serem avaliados, os hiperparâmetros a 
                     serem otimizados e outros parâmetros de 
                     treinamento, como o limite de outliers a ser usado
                     na filtragem dos dados.

  Se algum dos arquivos de configuração não puder ser lido, a função
  retorna None para todos os elementos da tupla.

  Parâmetros:
    verbose (bool): Se True, imprime mensagens detalhadas durante a 
                    leitura dos arquivos de configuração.

  Retorna:
    tuple: Uma tupla com os elementos configs_file_path, system_config, 
           applications_configs e training_config descritos 
           anteriormente.
  """

  # Primeiramente verifica se a variável de ambiente com o caminho da 
  # base do diretório com os arquivos dos scripts está definida,
  # o que ocorrerá somente se a variável de ambiente 
  # APPOPTIMIZER_BASE_FILES_PATH estiver definida ao executar o script
  # de otimização. 
  if BASE_FILES_PATH is None:
    # Se a varável de ambiente não estiver definida, imprime uma
    # mensagem de erro com o nome da variável de ambiente.
    print(f"❌ Variável de ambiente {BASE_FILES_PATH_ENV_NAME} com o caminho "
          "da base dos scripts não foi definida")

    # Como não temos como ler os arquivos de configuração pois não
    # sabemos o diretório com as configurações, retorna None para todos
    # os elementos da tupla.
    return None, None, None, None
  else:
    # Como o diretório base dos arquivos dos scripts está definido,
    # descobrimos o caminho do diretório com os arquivos de
    # configuração, que será composto por este diretório base e o
    # subdiretório definido na variável CONFIGS_FILES_DIR (este nome de
    # subdiretório é definido pela variável de ambiente 
    # APPOPTIMIZER_CONFIGS_DIR, que pode ser definida pelo adminstrador
    # do sistema para mudar o valor default "configs" usado quando a
    # variável não for definida.
    configs_file_path = BASE_FILES_PATH / CONFIGS_FILES_DIR

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
    system_config_file_path
  )

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
      configs_file_path / system_config['applications_path']
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

  # Define o caminho do arquivo de configuração com as informações
  # referentes ao treinamento dos modelos, que é composto pelo diretório 
  # de configuração e o nome do arquivo de configuração dos 
  # treinamentos, que é definido como "training_config.json". O nome
  # está atualmente fixo no código, mas podemos no futuro permitir que
  # seja definido por uma variável de ambiente ou no arquivo de
  # configuração do sistema.
  training_config_file_path = configs_file_path / 'training_config.json'

  # Lê as configurações usadas pelo treinamento dos modelos, usando a 
  # classe ReadTrainingConfig, que lê o arquivo de configuração com as
  # configurações de treinamento, como os modelos a serem avaliados, os
  # hiperparâmetros a serem otimizados e outros parâmetros de
  # treinamento, como o limite de outliers a ser usado na filtragem dos
  # dados. A função read_training_config desta classe retorna um
  # dicionário com as configurações lidas, caso o arquivo possa ser lido
  # sem erros. Se algum erro ocorrer durante a leitura do arquivo de
  # configuração de treinamento, a função retorna None.
  training_config = ReadTrainingConfig(
    verbose
  ).read_training_config(training_config_file_path)

  # Retorna uma tupla com cinco elementos, sendo o primeiro o caminho do
  # diretório de configuração, o segundo o dicionário com as
  # configurações do sistema, o terceiro o dicionário com as 
  # configurações das aplicações e o quarto o dicionário com as
  # configurações de treinamento. Se algum dos arquivos de configuração 
  # não pode ser lido, a função retorna None para o arquivo que não foi
  # lido.
  return (configs_file_path, system_config, applications_configs, 
          training_config)

def process_script_args():
  """
  Função para processar os argumentos passados para o script de 
  treinamento. A linha de comando do script deve ser composta pelos
  parâmetros do script, que são opcionais e atualmente temos somente o
  parâmetro "-v" ou "--verbose" para habilitar a verbosidade, e pelo 
  comando a ser executado. O comando a ser executado é obrigatório e 
  pode ser um dos seguintes:

  - applications: lista todas as aplicações que podemos treinar os 
                  modelos e que poderão posteriormente serem 
                  utilizadas pelos usuáriospelo script de otimização 
                  para gerar as sugestões de configuração de execução
                  das aplicações.
  - models: lista os modelos que são avaliados quando formos escolher o 
             melhor modelo para gerar os preditores de uma das
             aplicações.
  - train app1, app2, ..., appn: Faz todo o processo de treinamento, da 
                                 leitura e filtragem dos dados,
                                 otimização dos hiperparâmetros dos 
                                 modelos, escolha do melhor modelo e 
                                 treinamento deste melhor modelo com
                                 todos os dados, sendo gerado um modelo
                                 para auxiliar a geração das sugestões e
                                 outro, opcional, para predizer o tempo. 
                                 Cada aplicação da lista é considerada
                                 na ordem dada e os treinamentos são
                                 independentes.

  No momento, o script de treinamento possui somente dois parâmetros 
  opcionais:
   
  -v, --verbose: Habilita a verbosidade com informações sobre a melhor
                 sugestão de configuração escolhida para a aplicação.
  -h, --help: Mostra a ajuda com as opções do script de treinamento e
              com a sintaxe da linha de comando.

	Parâmetros:
    Não tem.

  Retorna:
    tuple: Uma tupla com os elementos command, verbose e parser, sendo o
           elemento command uma lista com o comando a ser executado e
           seus parâmetros se forem necessários (no momento, somente o
           comando train tem parâmetros, as aplicações para as quais
           treinaremos o preditor), o elemento verbose um booleano 
           indicando se a verbosidade está ou não habilitada e o 
           elemento parser uma referência para o objeto ArgumentParser 
           que processou os argumentos da linha de comando.                                                 
  """

  # Inicializa o analisador dos parâmetros do script de treinamento, com
  # uma descrição do que o script faz (parâmetro "description"), uma
  # descrição do formato da linha de comando do script (parâmetro
  # "usage"). Também informamos que vamos definir uma ajuda (help)
  # customizada, necessário para escrever a ajuda em português
  # (parâmetro "add_help" com o valor False), e uma classe de
  # formatação, necessária para lidar com múltiplas linhas e,
  # no nosso caso, que definimos um objeto da classe customizada 
  # CustomFormatter, mudar o nome na drescição da linha de comando do
  # inglês "usage:" para o português "uso" (parâmetro "formatter_class"
  # com a referência para a classe CustomFormatter customizada para
  # mostrar o "uso"). 
  parser = argparse.ArgumentParser(
    description="Script para treinar os modelos para todas as aplicações cujo "
                "uso será otimizado.", 
    usage="trainer [opções] Command [Parâmetros]",
    add_help=False, 
    formatter_class=CustomFormatter
  )

  # Cria um objeto cuja referência será armazenada em opcoes para 
  # agrupar as opções gerais do script, sendo que a string 
  # "Opções principais" passada ao criar o objeto será o texto mostrado
  # antes dos textos com as descrições de cada opção do script, com 
  # exceção da ajuda, que será colocada em um outro grupo único.
  opcoes = parser.add_argument_group("Opções principais")

  # Adiciona ao grupo de opções principais, referenciado por opcoes, o
  # parâmetro command, que é obrigatório e define o comando a ser 
  # executado. O parâmetro "type" com a classe str define que o 
  # parâmetro usa strings, o parâmetro "nargs" com a string "*" indica 
  # que o parâmetro recebe zero ou mais strings (devido ao "type" ser 
  # "str"), e o parâmetro "help" define uma descrição para a opção. 
  opcoes.add_argument(
    "command", 
    type=str, 
    nargs='*', 
    help=textwrap.dedent(
      """\
        Comando a ser executado. Pode ser um dos seguintes comandos:

        applications: lista todas as aplicações que podemos treinar os modelos.
        models: lista os modelos que são avaliados quando os preditores forem 
                gerados.

        train app1, app2, ..., appn: Faz todo o processo de treinamento, da 
                                      leitura e filtragem dos dados, otimização 
                                      dos hiperparâmetros dos modelos, escolha 
                                      do melhor modelo e treinamento deste 
                                      melhor modelo com todos os dados, sendo 
                                      gerado um modelo para auxiliar a geração 
                                      das sugestões e outro para predizer o 
                                      tempo.

         Cada aplicação da lista é considerada na ordem dada e os treinamentos 
         são idependentes, ou seja, passar a lista é equivalente a executar o 
         script com o comando para cada aplicação isoladamente.
      """
    )
  )

  # Adiciona ao grupo de opções gerais, referenciado por opcoes, a opção
  # -v ou --verbose que habilita a verbosidade do script de treinamento,
  # mostrando diversas informações sobre o processo de treinamento dos
  # modelos e a escoha do melhor modelo para gerar os preditores.
  opcoes.add_argument(
    "-v", "--verbose", 
    action="store_true", 
    default=False, 
    help="Habilita a verbosidade do script."
  )

  # Cria um objeto cuja referência será armazenada em ajuda para criar 
  # um grupo somente com a opção de ajuda, sendo que a string  "Ajuda" 
  # passada ao criar o objeto será o texto mostrado antes do texto da 
  # opção de ajuda.
  ajuda = parser.add_argument_group("Ajuda")

  # Adiciona ao grupo de ajuda, referenciado por ajuda, a opção -h ou
  # --help que mostra a ajuda do script de otimização, com a descrição
  # da linha de comando e quais são as opções disponíveis e como elas
  # devem ser usadas. O parâmetro "action" com a string "help" define 
  # que a opção é uma opção de ajuda, e o parâmetro "help" define uma 
  # descrição para a opção.
  ajuda.add_argument(
    "-h", "--help", 
    action="help", 
    help="Mostra esta mensagem de ajuda e sai."
  )

  # Processa os parâmetros passados ao script de treinamento, que inclui
  # os possíveis comandos descritos anteriormente (applications, models 
  # e train) e os opções do script de treinamento, a opção de 
  # verbosidade e a opção de ajuda.
  args = parser.parse_args()

  # Define se a verbosidade está habilitada ou não, com base no valor do 
  # parâmetro verbose, cujo valor é True se a opção -v ou --verbose foi 
  # passada na linha de comando, e False caso contrário. 
  verbose = args.verbose

  # Define o comando a ser executado, que é uma lista de strings, sendo 
  # a primeira string o comando a ser executado (applications, models ou
  # train) e as demais strings, se existirem, os parâmetros do comando.
  command = args.command

  # Retorna uma tupla com três elementos, sendo o primeiro a lista com o
  # comando a ser executado e seus parâmetros, o segundo um booleano
  # indicando se a verbosidade está habilitada ou não e o terceiro uma
  # referência para o objeto ArgumentParser que processou os argumentos
  # da linha de comando.
  return command, verbose, parser

def applications_command(applications_config):
  """
  Fumção para executar o comando "applications", que lista todas as 
  aplicações que podemos treinar os modelos para gerar o preditor, e que 
  poderão posteriormente serem utilizadas pelos usuários pelo script de 
  otimização para gerar as sugestões de configuração que permitirão 
  otimizar a execução das aplicações.  

  Parâmetros:
    applications_config (dict): Dicionário com todas as configurações
                                das aplicações que podem ser otimizadas
                                pelo script de otimização, sendo que
                                cada chave, associada e identificando
                                uma aplicação, tem uma referência para o
                                dicionário com as configurações dessa
                                aplicação.

  Retorna:
    bool: Sempre True, pois o comando sempre deveria ser executado com
          sucesso, já que applications_config é um dicionário e o nome
          que precisa ser usado no train é o nome interno da aplicação.
  """

  # Lista todas as aplicações que podem ter os modelos treinados para 
  # gerar o preditor. Para isso, como cada aplicação é uma chave do 
  # dicionário applications_config, percorremos todas as chaves desse 
  # dicionário. 
  for application_name in sorted(applications_config.keys()):
    # Para cada chave application_name do dicionário 
    # applications_config, que é o nome interno da aplicação, imprimimos
    # esse nome no terminal.
    print(f"➡️ {application_name}")

  # Como o comando applications sempre deve ser executado com sucesso,
  # retornamos True.
  return True

def models_command(training_config):
  """
  Função para executar o comando "models", que lista todos os modelos
  que podem ser treinados.

  Parâmetros:
    training_config (dict): Dicionário com todas as configurações
                            de treinamento. Como os dicionários com as
                            configurações de treinamento dos modelos são 
                            armazenados no diiconário cuja referência é
                            definida pela chave "models", então os nomes
                            dos modelos que podem ser treinados são as
                            chaves desse dicionário, logo basta listar 
                            as chaves para podermos mostrar os modelos
                            que serão usados ao construir o preditor de
                            uma aplicação.

  Retorna:
    bool: True, se o comando foi executado com sucesso, o que somente 
          não ocorrerá se a chave "models" não existir no dicionário 
          training_config, e False caso contrário, ou seja, se o acesso
          à chave gerar uma exceção KeyError. 
  """

  # Usa um bloco try para garantir que a chave "models" existe no 
  # dicionário training_config, e caso não exista, captura a exceção 
  # KeyError.
  try:
    # Percorre todas as chaves do dicionário definido pela chave 
    # "models" do dicionário training_config, que são os nomes dos 
    # modelos que podem ser treinados.
    for model_name in training_config['models'].keys():
      # Para cada chave model_name do dicionário definido pela chave 
      # "models" do dicionário training_config, que é o nome do modelo, 
      # imprimimos esse nome no terminal.
      print(f"➡️ {model_name}")

  # Captura a exceção KeyError caso a chave "models" não exista no
  # dicionário training_config.    
  except KeyError as e:
    print(f"❌ A chave '{e.args[0]}' não foi encontrada no arquivo de "
          "configuração de treinamento, logo não foi possível listar os "
          "modelos!")
    print(f"❌ Mensagem de erro: {str(e)}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Retorna False, pois o comando não pôde ser executado com sucesso.
    return False

  # Retorna True, pois o comando foi executado com sucesso.
  return True

def read_predictor_info_config(system_config, verbose=False):
  """
  Função para ler o arquivo de configuração com as informações dos
  preditores. No arquivo de configuração, e no dicionário que é obtido
  a partir da leitura do arquivo, a chave é o nome interno da aplicação,
  e a refência desta chave é para uma string com o nome do arquivo do
  preditor. Somente as aplicações que estão neste arquivo podem ter as
  suas execuções otimizadas pelo script de otimização.

  Parâmetros:
    system_config (dict): Dicionário com as configurações gerais
                          relacionadas aos scripts de otrimização e de
                          treinamento.
    verbose (bool): Indica se a verbosidade está habilitada, ou seja, se
                    deverão ser impressas as mensagens de log.

  Retorna:
    tuple: Uma tupla com três elementos, sendo o primeiro elemento o 
           caminho do diretório com os arquivos dos preditores e o
           arquivo com a configuração dos preditores, o segundo elemento
           um dicionário com as informações dos preditores, e o terceiro
           elemento uma referência para o objeto da classe
           PredictorsInfoConfig, que será necessário para atualizar o 
           arquivo de configuração com as informações dos preditores. Se
           algum erro ocorrer durante a leitura do arquivo de
           configuração dos preditores, a função retorna None para o
           segundo elemento da tupla, o dicionário que deveria armazenar
           as aplicaççoes como chaves e os nomes dos arquivos dos 
           preditores como valores dessas chaves.
  """
  
  # Define o caminho do diretório com os arquivos dos preditores e do 
  # arquivo de configuração com qual preditor é o melhor para cada
  # aplicação.
  predictors_file_path = (BASE_FILES_PATH / 
                          Path(system_config['predictors_path']))
  
  # Define o caminho do arquivo de configuração com as informações dos
  # preditores, que contém uma entrada para cada aplicação com o nome do
  # preditor que foi treinado para essa aplicação e que será o usado ao
  # gerar as sugestões de configuração de execução da aplicação.
  predictors_info_file_path = (
    predictors_file_path / system_config["predictors_info_config_filename"]
  )

  # Cria um objeto da classe PredictorsInfoConfig, que será usado para
  # ler o arquivo de configuração com as informações dos preditores e
  # para atualizar esse arquivo com as informações do preditor que foi
  # treinado para cada aplicação.
  predictors_info_config_obj = PredictorsInfoConfig()

  # Lê o arquivo de configuração com as informações dos preditores, que
  # é um arquivo JSON, e retorna um dicionário com as informações dos
  # preditores, caso o arquivo possa ser lido sem erros. Se algum erro
  # ocorrer durante a leitura do arquivo de configuração dos preditores,
  # a função retorna None.
  predictors_info_config = (
      predictors_info_config_obj.read_predictors_info_config(
          predictors_info_file_path
      )
  )

  # Se a verbosidade está habilitada, imprimimos as informações dos
  # preditores lidas do arquivo de configuração.
  if verbose:
    # Imprime uma mensagem de informação indicando cada aplicação e o
    # nome do arquivo com o preditor que foi treinado para essa
    # aplicação.
    print("➡️ Informações dos preditores lidas do arquivo de configuração:")

    # Varre a lista de preditores e imprime as informações para cada
    # aplicação que pode ter a sua execução otimizada e para a qual portanto
    # será treidado os modelos para gerar o preditor.
    for application_name, predictor_file_name in predictors_info_config.items():
      # Imprime o nome da aplicação e o nome do arquivo com o preditor
      # que foi treinado para essa aplicação.
      print(f"➖➡️ Aplicação: {application_name}; "
            f"Nome do arquivo com o preditor treinado: {predictor_file_name}")
      
    # Imprime uma linha em branco para separar as informações dos
    # preditores dos outros logs.
    print("")

  # Retorna uma tupla com três elementos, sendo o primeiro o caminho do
  # diretório com os arquivos dos preditores e o arquivo com a
  # configuração dos preditores, o segundo um dicionário com as
  # informações dos preditores, e o terceiro elemento uma referência
  # para o objeto usado ao ler o arquivo de configuração com as
  # informações dos preditores.
  return (predictors_file_path, predictors_info_config, 
          predictors_info_config_obj)

def read_training_data(dataset_files, system_config, used_columns, 
                       verbose=False):
  """
  Função para ler os dados de treinamento de uma aplicação, usando as
  informações da aplicação e do sistema.

  Parâmetros:
    dataset_files (list): Lista com os nomes dos arquivos com os 
                          conjuntos de dados a serem lidos e
                          concatenados.
    system_config (dict): Dicionário com as configurações gerais
                          relacionadas aos scripts de otrimização e de
                          treinamento. No caso da função, iremos usar
                          chave 'dataset_path' com o nome do diretório 
                          em que estão armazenados todos os conjuntos de
                          dados das aplicações usados nos treinamentos.
    used_columns (list): Lista com os nomes das colunas a serem lidas
                         dos arquivos con os conjuntos de dados das
                         aplicações. Os nomes das colunas vão depender
                         da aplicação.
    verbose (bool): Indica se a verbosidade está habilitada, ou seja, se
                    deverão ser impressas as mensagens de log.

  Retorna:
    DataFrame | None: Se nenhum erro ocorreu, os conjuntos de dados da 
                      aplicação, para a qual estamos gerando o preditor, 
                      lidos e concatenados em um único conjunto de
                      dados, ou None caso algum erro ocorra durante a 
                      leitura dos arquivos de dados.
  """  

  try:
    # Inicializa um DataFrame vazio que será usado para armazenar as
    # colunas relevantes (passadas em used_columns) dos arquivos com os 
    # conjuntos de dados das aplicações.
    dataset = pd.DataFrame()

    # Percorre todos os arquivos com os conjuntos de dados das
    # aplicações, que são passados na lista dataset_files.
    for file_name in dataset_files:
      # Define o caminho completo do arquivo com o conjunto de dados da
      # aplicação, que é composto pelo diretório base dos arquivos dos
      # scripts, pelo subdiretório definido na chave 'dataset_path' do
      # dicionário system_config e pelo nome do arquivo com o conjunto
      # de dados da aplicação.
      full_file_path = (
        BASE_FILES_PATH / Path(system_config['dataset_path']) /  file_name
      )

      # Lê o arquivo com o conjunto de dados da aplicação, usando
      # somente as colunas relevantes passadas em used_columns, e
      # armazena o resultado em um DataFrame temporário.
      file_dataset = pd.read_csv(full_file_path, usecols=used_columns)
      
      # Concatena o DataFrame temporário com o DataFrame principal, que
      # irá armazenar todos os conjuntos de dados das aplicações, lidos
      # e concatenados em um único conjunto de dados.
      dataset = pd.concat([dataset, file_dataset])

    # Como os índices dos DataFrames lidos podem ser diferentes, e como
    # o concatenarmos os DataFrames, os índices podem ficar desordenados
    # e podem haver índices duplicados, então redefinimos os índices do
    # DataFrame principal.
    dataset = dataset.reset_index(drop=True)		

    # Se a verbosidade está habilitada, imprimimos o conjunto de dados
    # original da aplicação e os dados estatísticos referentes a esse
    # conjunto de dados.
    if verbose:
      # Imprime o conjunto de dados original da aplicação final, já com
      # todos os conjuntos de dados concatenados.
      print("➖➡️ Conjunto de dados original da aplicação:")
      print("\n", dataset.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", 
            sep="")

      # Imprime os dados estatísticos referentes a esse conjunto de
      # dados.
      print("➖➖➡️ Dados estatísticos referentes ao conjunto de dados "
            "original:")
      print("\n", dataset.describe().to_markdown(tablefmt="grid", 
                                                 floatfmt=".2f"), "\n", sep="")
      
    # Como conseguimos ler o(s) conjunto(s) de dados, então retornamos o
    # DataFrame com esse(s) conjunto(s).
    return dataset

  # O arquivo não foi encontrado.
  except FileNotFoundError as e:
    print(f"❌ O arquivo {full_file_path.name} não foi encontrado!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Retorna None porque o(s) conjunto(s) de dados não foi(foram) lidos
    # com sucesso.
    return None

  # O caminho é para um diretório e não um arquivo.
  except IsADirectoryError as e:
    print(f"❌ O caminho {full_file_path.name} é um diretório!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Retorna None porque o(s) conjunto(s) de dados não foi(foram) lidos
    # com sucesso.
    return None

  # O arquivo não pôde ser aberto devido a problemas de permissão.
  except PermissionError as e:
    print(f"❌ Erro de permissão ao acessar o arquivo {full_file_path.name}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Retorna None porque o(s) conjunto(s) de dados não foi(foram) lidos
    # com sucesso.
    return None

  # Ocorreu um erro de I/O ao acessar o arquivo.
  except IOError as e:
    print(f"❌ Erro de I/O ao acessar o arquivo {full_file_path.name}!")
    print(f"❌ Código do erro: {e.errno}; Mensagem: {e.strerror}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Retorna None porque o(s) conjunto(s) de dados não foi(foram) lidos
    # com sucesso.
    return None

  # Ocorreu um erro de codigicação dos caracteres do arquivo, por 
  # exemplo, o arquivo está no formato Latin-1 ao invés do UTF-8.
  except UnicodeDecodeError as e:
    print("❌ Erro de codificação ao acessar o arquivo "
          f"{full_file_path.name}!")
    print(f"❌ Parâmetros do erro: {e.args}!")
    print(f"❌ Mensagem de erro: {str(e)}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Retorna None porque o(s) conjunto(s) de dados não foi(foram) lidos
    # com sucesso.
    return None

  # Algumas das colunas especificadas em used_columns não existem no
  # arquivo.
  except ValueError as e:
    print(f"❌ Erro ao processar os dados do arquivo {full_file_path.name}, "
          "um valor errado foi usado!")
    print(f"❌ Parâmetros do erro: {e.args}!")
    print(f"❌ Mensagem de erro: {str(e)}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Retorna None porque o(s) conjunto(s) de dados não foi(foram) lidos
    # com sucesso.
    return None

  # Ocorreu alguma outra exceção inesperada.
  except Exception as e:
    print(f"❌ Erro desconhecido ao acessar o arquivo {full_file_path.name}!")
    print(f"❌ Mensagem de erro: {str(e)}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Retorna None porque o(s) conjunto(s) de dados não foi(foram) lidos
    # com sucesso.
    return None

def filter_training_data(dataset, input_variables, filter_variables, 
                         outlier_limit, verbose=False):
  """
  Função para filtrar os dados de treinamento de uma aplicação, usando as
  informações da aplicação e do sistema. A filtragem será feita pela
  função filter de um objeto da classe Filter, usando as variáveis de
  filtragem, dadas na lista filter_variables, e o limite definido a 
  partir do valor em ponto flutuante definido em outlier_limit. A 
  filtragem, para cada uma dessas variáveis, será feita usando o desvio
  absoluto em relação à mediana dos valores dessa variável para as 
  repetições de cada teste e o valor dado em outlier_limit.

  Parâmetros:
    dataset (DataFrame): Dados de treinamento.
    input_variables (list[str]): Lista com os nomes das variáveis de 
                                 entrada.
    filter_variables (list[str]): Lista com os nomes das variáveis de
                                  filtro.
    outlier_limit (float): Limite para identificação de outliers.
    verbose (bool): Indica se a verbosidade está habilitada, ou seja, se
                    deverão ser impressas as mensagens de log.

  Retorna:
    DataFrame | None : O conjunto de dados da aplicação com os outliers 
                       filtrados, ou seja, com os outliers removidos se
                       o processo de filtragem for bem sucedido, ou None
                       caso algum erro ocorra durante o processo de
                       filtragem dos dados.
  """

  try:
    # Cria um objeto da classe FilterOutliers, que será usado para 
    # filtrar os outliers do conjunto de dados da aplicação.
    dataset_filter = FilterOutliers()

    # Filtra os outliers do conjunto de dados da aplicação dataset,
    # usando as variáveis de entrada, as variáveis de filtro e o limite
    # para identificação de outliers. O resultado é um novo conjunto de
    # dados da aplicação, com os outliers removidos, que tem que a sua
    # referência armazenada na variável clean_dataset.
    clean_dataset = dataset_filter.filter(dataset, input_variables, 
                                          filter_variables, outlier_limit)

    # Se a verbosidade está habilitada, imprimimos o conjunto de dados
    # da aplicação após a filtragem dos outliers e os dados estatísticos
    # referentes a esse conjunto de dados filtrado.
    if verbose:
      # Imprime o conjunto de dados da aplicação após a filtragem dos
      # outliers.
      print("➖➡️ Conjunto de dados da aplicação após a filtragem dos "
            "outliers:")
      print("\n", clean_dataset.to_markdown(tablefmt="grid", floatfmt=".2f"), 
            "\n", sep="")

      # Imprime os dados estatísticos referentes a esse conjunto de 
      # dados filtrado.
      print("➖➖➡️ Dados estatísticos referentes ao conjunto de dados "
            "filtrado:")
      print("\n", clean_dataset.describe().to_markdown(tablefmt="grid", 
                                                      floatfmt=".2f"), "\n", 
            sep="")

    # Retorna uma referência para o conjunto de dados da aplicação com
    # os outliers filtrados.
    return clean_dataset

  # Foi passado um tipo inválido em um dos parâmetros da função filter.
  except ValueError as e:
    print("❌ Erro ao filtrar os outliers do conjunto de dados da "
          "aplicação, um valor errado foi usado!")
    print(f"❌ Parâmetros do erro: {e.args}!")
    print(f"❌ Mensagem do erro: {str(e)}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como não conseguimos filtrar o conjunto de dados com sucesso, 
    # retorna None.
    return None

  # Em um dos parâmetros de variáveis da funcão filter, foi passada uma
  # variável que não existe no conjunto de dados da aplicação.
  except KeyError as e:
    print("❌ Erro ao filtrar os outliers do conjunto de dados da "
          "aplicação, uma chave errada foi usada!")
    print(f"❌ Parâmetros do erro: {e.args}!")
    print(f"❌ Mensagem do erro: {str(e)}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como não conseguimos filtrar o conjunto de dados com sucesso, 
    # retorna None.
    return None

  # Ocorreu alguma outra exceção inesperada.
  except Exception as e:
    print("❌ Erro desconhecido ao filtrar os outliers do conjunto de dados "
          "da aplicação!")
    print(f"❌ Mensagem do erro: {str(e)}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como não conseguimos filtrar o conjunto de dados com sucesso, 
    # retorna None.
    return None

def optimize_hyperparameters(clean_dataset, models, 
                             suggestion_variables,
                             application_variables,
                             group_variables,
                             suggestion_prediction_variable,
                             verbose=False): 
  """
  Função para otimizar os hiperparâmetros dos modelos que serão 
  considerados e terão os seus hiperparâmetros selecionados otimizados
  antes do melhor modelo, escolhido pela validação cruzada, ser usado
  para construir o preditor, composto pelo estimador auxiliar cujas
  predições serão usadas para o script de otimização escolher a melhor
  sugestão de configuração e, se habilitado, o estimador para predizer o
  tempo de execução ao executar a aplicação com essa eguestão. A escolha
  dos melhores valores para os hiperparâmetros selecionados de cada
  modelo será feita pela classe BestHiperparams, que internatemtne usará
  na função optimize um objeto da classe GridSearchCV do scikit-learn
  para otimizar os hiperparãmetros de um modelo. Como a classe 
  GridSearchCV usa internamente uma validação cruzada, será usada a
  validação LOGO com os grupos definidos pelas variáveis dadas na
  lista group_variables. As variáveis da aplicação, dadas na
  lista application_variables, e as relacionadas às componentes de cada
  configuação que define uma sugestão, dadas na lista 
  suggestion_variables, serão usadas para treinar os modelos da
  validação cruzada, cujo conjunto de dados, que inclui colunas para 
  todas as variáveis definidas pelas listas suggestion_variables, 
  application_variables, group_variables e
  uggestion_prediction_variable, irá predizer a variável dada em
  suggestion_prediction_variable (no nosso caso, atualmente a variável,
  para todas as aplicações, é a EDP).
    
  Parâmetros:
    models (dict): Dicionário em que cada chava é o nome interno do 
                   modelo, por exemplo ExtraTressRegressor,e o conteúdo
                   da chave é também um dicionário em que cada chave é
                   um dos hiperparâmetros do modelo associado à chave, e
                   o conteúdo é os valores dos hiperparâmetros que serão
                   avaliados.
    clean_dataset (DataFrame): Conjunto de dados já filtrado com os
                               testes para a aplicação para a qual 
                               desejamos otimizar os hiperparâmetros dos
                               modelos que serão avaliados, que será
                               usado em todos os treinamendos da
                               validação cruzada, usando o LOGO, feita
                               quando cada combinação de valores de
                               hiperparâmetros é avaliada para cada um
                               dos modelos definidos pelas chaves do
                               dicionário models.
    suggestion_variables (list[str]): Lista com os nomes das variáveis
                                      que definem as componentes de uma
                                      sugestão de configuração.
    application_variables (list[str]): Lista com os nomes das variáveis
                                       da aplicação, mas mão exatamente
                                       as originais passadas pelos 
                                       usuários, mas as que foram 
                                       convertidas, sendo as originais
                                       usadas somente se nenhuma
                                       conversão foi efetivamente
                                       feita (a conversão "copy" no
                                       script de otimização). Por 
                                       exemplo, no RAxML, não foi usado 
                                       o nome do arquivo, mas o tamanho
                                       desse arquivo.
    group_variavbles (list[str]): Lista com as variáveis, em geral as
                                  variáveis da aplicação definidas pelo
                                  usuário ou convertidas, que definem os
                                  grupos usados pela validação LOGO.
    suggestion_prediction_variable (str): Nome da variável alvo dos
                                          modelos treinados para avaliar
                                          as possíveis combinações de
                                          hiperparâmetros para cada
                                          modelo, usando a validação
                                          cruzada e o LOGO. Esta 
                                          variável é a mesma usada pelo
                                          modelo auxiliar usado pelo
                                          script de otimização ao 
                                          escolher a melhor sugestão de
                                          comfiguração. Nas aplicações
                                          que estamos avaliando, usamos
                                          a variável EDP.                                                              
    verbose (bool): Indica se a verbosidade está habilitada, ou seja, se
                    deverão ser impressas as mensagens de log.

    Retorna:
      tuple: Uma tupla com dois elementos. Se não ocorrerem erros, o 
             primeiro elemento da tupla é um Diciońario em que cada 
             chave é o indentificador interno do modelo e a referência
             o objeto do modelo representado pela chave já inicializado,
             e o segundo elemento é também um dicionário, também 
             indexado pelo nome interno do modelo, com os melhores
             hiperparâmetros para este modelo. Se algum erro ocorrer,
             retorna None para ambos os elementos.
  """

  # Inicializa o dicionário com os modelos já inicialiados com os 
  # melhores hiperoarâmetros. A chave será o nome interno do modelo que
  # são as chaves do dicionário referenciado pela chabe 'models" do 
  # arquivo de configuração do treinamento e a referência da chave será
  # o modelo inicialozado com os melhores valores para os
  # hiperparâmetros otimizados.
  initialized_models = {}

  # Inicializa o dicionário com os melhores valores dos hiperparâmetros
  # para cada modelo. A chave é, como no dicionário initialized_models,
  # a chave do dicionário referenciado pela chave 'models' do arquivo de
  # configuração de treinamento,
  best_params_models = {}

  try:
    # Vamos executar a otimização dos hiperparâmetros para cada modelo
    # definido em models, que é a mesma referência ao dicionário 
    # definido pela chave 'models' do dicionário de configuração do 
    # treinamento que foi lido do arquivo de configuração ao inicializar
    # o script de treinamento. Para cada entrada do dicionário models,
    # sendo a chave aramzenada em preditor_key, o nome interno do 
    # modelo, e model_info o dicionário com as informações para o uso
    # e otimização dos parâmetros deste modelo.
    for preditor_key, model_info in models.items():
      # Se a verbosidade estiver habilitada, mostra o modelo com o nome 
      # interno preditor_key e o nome definido pela chave 'name' do
      # dicionário models_info, e os hiperparâmetros que serão 
      # otimizados e os possíveis valores considerados para cada 
      # hiperparâmetro, que são definidos no dicionário definido pela
      # chave 'grid_search_parms' de model_info, sendo cada chave um dos
      # hiperparâmetros a serem otimizados e a referência desta chave
      # a lista com os possíveis valores a serem consideramdos para esse
      # hiperparâmetro.
      if verbose:
        print(f"➖➡️ Otimizando o modelo {model_info['name']} usando os "
              f"hiperparâmetros {model_info['grid_search_parms']} e a busca "
              "em  grade, usando a variável alvo "
              f"{suggestion_prediction_variable}:")
        
      # Descobre e importa o modelo com o nome interno preditor_key, 
      # de modo dinâmico, sendo que o  caminho para importar esse
      # modelo é definido na chave 'import_path' do dicionário 
      # model_info. Como o caminho indica como o modelo seria acessado
      # diretamente, importando somente a bibliotea básica do Python que
      # está implementado o modelo, por exemplo, para o regressor Extra 
      # Trees do scikit-learn 'sklearn.ensemble.ExtraTreesRegressor', 
      # então usamos o rsplilt da classe string, para separar as strings
      # por um separador, no caso o ",", a partir da direita (o "r" em
      # rsplit), em duas strings de acordo com o último ponto, sendo o
      # primeiro, armazenado em module_path, o caminho que seria usado 
      # em um "from" para importar um objeto específico da biblioteca
      # (ou módulo), e o segundo, armazenado em model_name, o nome da 
      # classe do objeto. No caso das Extra Trees, module_path seria
      # uma referência para a string "sklearn.ensemble" e model_neme
      # seria uma referência para a string "ExtraTreesRegressor".
      module_path, model_name = model_info['import_path'].rsplit(".", 1)

      # Uma vez que definimos o caminho completo na biblioteca ou 
      # módulo, usamos a função import_module da biblioteca padrão do
      # Python importlib para dinâmicamente importar a biblioteca que
      # implementa o modelo, sendo a referência ao objeto que representa
      # a biblioteca armazenada em model_module.
      # If the module cannot be imported, import_module() will raise 
      # ImportError or an appropriate subclass like ModuleNotFoundError.
      model_module = importlib.import_module(module_path)

      # Uma vez que a biblioteca tenha sido importada em um objeto
      # referenciado por model_module, os seus atributos serão o que
      # podemos importar dessa bibliotca, ou seja, um desses atributos
      # deveria ser o nome daa classe do objeto que implementa o modelo
      # para o qual iremos otimizar os seus hiperparâmetros, supondo que
      # o nome passado na chave 'import_path' do dicionário model_info
      # está correto. Se estiver correto, retorna a referência para a
      # classe do modelo e, em caso contrário (se o atributo não 
      # existir, retorna None).
      model = getattr(model_module, model_name, None)

      # Verifica o objeto model_module tem o atributo model_name.
      if model is not None:
        # Como o objeto model mão é None, ou seja, como o objeto 
        # model_module tem o atributo model_name,  então vamos criar um
        # objeto da classe BestHiperparams, usada para otimizar os
        # hiperparâmetros de um  modelo, e armazenar a rederência a esse 
        # objeto em best_hyper.
        best_hyper = BestHiperparams(verbose=DEBUG_CODE)

        # Chamamos a função optimize do objeto best_hyper para otimizar
        # os hiperparâmetros do modelo que estamos avaliando. Para a
        # função são passados o conjunto de dados fultrado 
        # (clean_dataset), as variáveis suggestion_variables associadas
        # as sugestões das configurações, as variáveis 
        # application_variables associadas aos parâmetros da aplicação
        # passados pelos usuários e convertidos quando necessário, as
        # variáveis group_variables usadas para definir os grupos usados
        # pela validação LOGO usada internamente ao escolher a melhor
        # combinação para os valores dos hiperparâmtros, a variável
        # suggestion_prediction_variable que define a variável alvo 
        # usadas nos treinamentos feitos ao verificar cada combinação
        # para os valores dos hiperparâmetros (nos testes que fizemos,
        # usamos a EDP), e o modelo inicializado. Como o obejo model é
        # uma referência para a classe do modelo que estamos otimizando,
        # podemos usar model como se fosse usada ao criar um objeto da
        # classe, como em model() se não definimos hiperparâmetros fixos
        # para o modelo que não veriam durante a escolha dos melhores
        # valores para os hiperparâmetros que variam, ou seja, se 
        # model_info não tem a chave 'fixed_params' e 
        # model(**model_info['fixed_params']) se essa chave existir 
        # (como estamos fazendo uma descompactação do dicionário, cada 
        # entrada, definida por uma chave x como o nome do
        # hiperparâmerro x e o valor y como o valor desse hiperparâmetro
        # para x-y.), e finalmente, o dicionário com os hiperparâmetros
        # do modelo a serem otimizados, sendo a chave um dos
        # hiperparâmetros e o seu valor uma lista com os valores a serem
        # avaliados para esse hiperparâmetro. A função retorna um
        # dicionário, cuja referência é aemazenada best_params, em que
        # cada chave é um hiperparâmtro avaliado e a referência dessa
        # chave o melhor valor escolhido para esse hiperparâmetro, e a
        # melhor pontuação associada os melhores valores para os
        # hiperparâmetros, armazenada em best_score. A função retorna
        # as seguintes referências:
        # 
        # best_params: dicionário em que a chave é um dos 
        #              hiperparâmetros e o valor o melhor valor para
        #              esse hiperparâmetro. 
        # best_score: Pontuação para os melhores valores dos 
        #             hiperparâmetros. A pontuação usada é a acurácia.
        best_params, best_score = best_hyper.optimize(
          clean_dataset, 
          suggestion_variables,
          application_variables,
          group_variables,
          suggestion_prediction_variable, 
          model(**model_info['fixed_params']) if 'fixed_params' in model_info 
                                              else model(),
          model_info['grid_search_parms']
        )

        # Se a verbosidade estiver habilitada, imperime informações 
        # referenteas a otimização dos hiperparâmetros do
        if verbose:
          # Imprime o nome do modelo, o dicionário com os melhores
          # valores para os hiperparâmetros avaliados e a pontuação 
          # associada aos melhores valores dos hiperparâmetros.         
          print(f"➖➖➡️  Modelo {model_info['name']}: Melhores "
                f"hiperparâmetros -> {best_params}; Melhor score -> "
                f"{best_score}")

          # Imprime o DataFrame com as informações referentes a 
          # otimização dos hiperparâmetros, com todas as informações
          # relevantes refrentes aos treinamentos feitos para escolher
          # os melhores valores para os hiperparâmetros. 
          print("➖➖➖➡️ Dataframe com a avaliação de todas as combinações "
                "dos hiperparâmetros:")         

          # Primeiramente obtem essas informações, que sáo retornadas
          # pela função get_hyperparams_scores do ojeto best_hyper em
          # um DataFrame do Pandas.
          hyperparams_score = best_hyper.get_hyperparams_scores() 

          # Imprime o DataFrame do Pandas obtido.
          print("\n", hyperparams_score.to_markdown(tablefmt="grid", 
                                                    floatfmt=".2f"), "\n", 
                sep="")

        # Verifica se a chave 'fixed_params' existe no dicionário
        # models_info, ou seja, se foram definidos hiperparâmerros com
        # valores ficos (nos nossos testes, usamos o hiperparâmetro
        # random_state=42).
        if 'fixed_params' in model_info:
            # Se a chave 'fixed_params' existe no dicionário model_info,
            # então ela é um dicionário em que a chave é um 
            # hiperparâmetro fixo e a referência da chave esse valor, 
            # então juntamos o dicionpario best_params com o dicionário
            # rferenciado pela chave 'fixed_params' do dicionário 
            # model_info e armazenados o resultado, ou seja, a refeência
            # ao novo dicionário, em best_params.
            best_params = dict(**best_params, **model_info['fixed_params'])

        # Adiciona ao dicionário dos modelos inicializados, na chave 
        # preditor_key com o nome interno do modelo, o modelo 
        # inicializado (lembrando que podemos usar o objeto model para
        # criar um objeto da classe referenciada por esse objeto), 
        # inicializado com os hiperparâmetros definidos no dicionário
        # best_params. Como estamos fazendo uma descompactação do
        # dicionário, cada entrada, definida por uma chave x como o nome
        # do hiperparâmerro x e o valor y como o valor desse 
        # hiperparâmetro para x-y.
        initialized_models[preditor_key] = model(**best_params) 

        # Armazana, na chave com o mesmo nome interno do model
        # opreditor_key, os hiperparâmerros com os valores otimizados,
        # assim como os hiperparâmetros com os valores fixos.
        best_params_models[preditor_key] = best_params
      else:
        # Como não foi encontrado uma classe com o nome model_name na
        # bibloreca referenciada por model_module, mostra um erro 
        # informando que o modelo não existe e, por coerência com o erro
        # do módulo não existir, geramos a exceção ImportError.
        print(f"❌ O caminho de importação {model_info['import_path']} é "
              f"inválido. A classe do modelo {model_name} não existe na "
              f"bibloteca referenciada por {module_path}!")
        raise ImportError(f"O modelo {model_name} não existe no módulo "
                          f"{module_path}!", name=module_path)

    # Retorna os dicionários com os modelos avaliados já inicializados e 
    # os melhores valores para os hiperparâmetros para os modelos 
    # avaliados, conjuntamente com os hiperparâmetros com os valores
    # fixos se foram definidos.
    return initialized_models, best_params_models

  # Foi passado um tipo inválido em um dos parâmetros da função
  # optimize.
  except ValueError as e:
    print("❌ Erro ao otimizar os hiperparâmetros dos modelos, um valor "
          "errado foi usado!")
    print(f"❌ Parâmetros do erro: {e.args}!")
    print(f"❌ Mensagem de erro: {str(e)}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como não foi possível otimizar os hiperparâmetros de algum dos
    # models, retorna None para os dois elementos da tupla.
    return None, None
  
  # Em um dos parâmetros de variáveis da função optimize, foi passada
  # uma variável que não existe no conjunto de dados da aplicação.
  except KeyError as e:
    print("❌ Erro ao otimizar os hiperparâmetros dos modelos, uma chave "
          "errada foi usada!")
    print(f"❌ Parâmetros do erro: {e.args}!")
    print(f"❌ Mensagem de erro: {str(e)}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como não foi possível otimizar os hiperparâmetros de algum dos
    # models, retorna None para os dois elementos da tupla.
    return None, None

  # Ocorreu um erro ao importar um dos modelos.
  except ImportError as e:
    print("❌ Erro ao otimizar os hiperparâmetros dos modelos, uma classe não "
          f"existe no módulo {e.name}!")
    print(f"❌ Mensagem de erro: {str(e)}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como não foi possível otimizar os hiperparâmetros de algum dos
    # models, retorna None para os dois elementos da tupla.
    return None, None

  # Ocorreu um erro ao importar um dos modelos.
  except ModuleNotFoundError as e:
    print("❌ Erro ao otimizar os hiperparâmetros dos modelos, uma biblioteca "
          f"{e.name} errada foi usada")
    print(f"❌ Mensagem de erro: {str(e)}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como não foi possível otimizar os hiperparâmetros de algum dos
    # models, retorna None para os dois elementos da tupla.
    return None, None
  
  # Ocorreu alguma outra exceção inesperada.
  except Exception as e:
    print("❌ Erro desconhecido ao otimizar os hiperparâmetros dos modelos!")
    print(f"❌ Mensagem de erro: {str(e)}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como não foi possível otimizar os hiperparâmetros de algum dos
    # models, retorna None para os dois elementos da tupla.
    return None, None
  
def choose_best_model(clean_dataset, initialized_models, suggestion_variables,
                      application_variables, group_variables, 
                      suggestion_prediction_variable, verbose=False): 
  """
  Função para escolher o melhor modelo dentre os modelos que serão 
  considerados, send este modelo o usado para construir o preditor, 
  composto pelo estimador auxiliar cujas predições serão usadas para o 
  script de otimização escolher a melhor sugestão de configuração e, se 
  habilitado, o estimador para predizer o tempo de execução ao executar
  a aplicação com essa suguestão. A escolha do melhor modelo será feita 
  pela classe DiscoverBestModel, que internatemente usará,
  na função best_model, a função cross_validate do scikit-learn
  para fazer a validação cruzada para cada um dos modelos, para depois
  escolher o melhor modelo, como a melhor pontuação. A validação
  usada será a LOGO com os grupos definidos pelas variáveis dadas na
  lista group_variables. As variáveis da aplicação, dadas na lista 
  application_variables, e as relacionadas às componentes de cada 
  configuação que define uma sugestão, dadas na lista 
  suggestion_variables, serão usadas para treinar os modelos da
  validação cruzada, cujo conjunto de dados, que inclui colunas para 
  todas as variáveis definidas pelas listas suggestion_variables, 
  application_variables, group_variables e 
  suggestion_prediction_variable, irá predizer a variável dada em 
  suggestion_prediction_variable (no nosso caso, atualmente a variável, 
  para todas as aplicações, é a EDP).
    

  Parâmetros:
    clean_dataset (DataFrame): Conjunto de dados já filtrado com os
                               testes para a aplicação para a qual 
                               desejamos escolher o melhor modelo dentre
                               os modelos que serão avaliados, que será
                               usado em todos os treinamendos da
                               validação cruzada, usando o LOGO, feita
                               quando as pontuações para cada um dos
                               modelos é calculada, para depois escolher
                               o modelo com a maior pontuação.
    initialized_models (dict): Dicionário em que cada chava é o nome
                               interno do modelo, por exemplo 
                               ExtraTressRegressor, e o conteúdo da 
                               chave é um objeto do modelo já 
                               inicializado e pronto para ser treinado,
                               com os melhores valores para os
                               hiperparâmetros otimizados e os
                               hiperparâmetros fixos.
    suggestion_variables (list[str]): Lista com os nomes das variáveis
                                      que definem as componentes de uma
                                      sugestão de configuração.
    application_variables (list[str]): Lista com os nomes das variáveis
                                       da aplicação, mas mão exatamente
                                       as originais passadas pelos 
                                       usuários, mas as que foram 
                                       convertidas, sendo as originais
                                       usadas somente se nenhuma
                                       conversão foi efetivamente
                                       feita (a conversão "copy" no
                                       script de otimização). Por 
                                       exemplo, no RAxML, não foi usado 
                                       o nome do arquivo, mas o tamanho
                                       desse arquivo.
    group_variavbles (list[str]): Lista com as variáveis, em geral as
                                  variáveis da aplicação definidas pelo
                                  usuário ou convertidas, que definem os
                                  grupos usados pela validação LOGO.
    suggestion_prediction_variable (str): Nome da variável alvo dos
                                          modelos treinados para avaliar
                                          as possíveis combinações de
                                          hiperparâmetros para cada
                                          modelo, usando a validação
                                          cruzada e o LOGO. Esta 
                                          variável é a mesma usada pelo
                                          modelo auxiliar usado pelo
                                          script de otimização ao 
                                          escolher a melhor sugestão de
                                          comfiguração. Nas aplicações
                                          que estamos avaliando, usamos
                                          a variável EDP.                                                              
    verbose (bool): Indica se a verbosidade está habilitada, ou seja, se
                    deverão ser impressas as mensagens de log.

  Retorna:
    tuple: Uma tupla com dois elementos. Se não ocorrerem erros, o 
           primeiro elemento da tupla é uma string com o nome interno do
           melhor modelo e o segundo elemento a pontuação dada a esse
           melhor modelo. Se algum erro ocorrer, retorna None para ambos
           os elementos.
  """

  try:
    # Se a verbosidade estiver habilitada, imprime uma mensagem 
    # mostrando todos os modelos que serão avaliados pela validação
    # cruzada cujos resultados para cada modelo serão usados para
    # determinar a sua pontuação e depois escolher o melhor modelo.
    if verbose:	
      print("➖➡️ Determinando o melhor modelo dentre os modelos da lista  "
            f"{', '.join(initialized_models.keys())}, usando a validação "
            "cruzada com o LOGOm e a variável alvo "
            f"{suggestion_prediction_variable}:")

    # Cria um objeto da classe DiscoverBestModel, que será usada para
    # descobriri o melhor modelo dentre os que foram inicializados e
    # armazenados no dicionário initialized_models, e armazena a 
    # referência ao objeto em discover_best_model.
    discover_best_model = DiscoverBestModel(verbose=DEBUG_CODE)

    # Usa a função 	best_model do objeto referenciado por 
    # discover_best_model para descobrir o melhor modelo dentre os 
    # definidos pelas chaves do dicionário initialized_models. A função
    # retorna quatro referências, duas que retornaremos, pois são usadas
    # fora da função, e duas que somente usaremos para a opção de 
    # verbosidade. A função retorna o seguinte:
    #
    # best_model_name: Nome interno do melhor modelo, sendo o nome uma
    #                  das chaves do dicionário initialized_models.
    # best_model_score: Pontuação do melhir nodelo. Como podemos ter
    #                   mais de uma pontuação, é um dicionário em que
    #                   cada chave é uma das pontuações e o valor e
    #                   dessa pontuação. No nossos testes temos duas
    #                   pontuações, a acurácia e o negativo da 
    #                   diferença, sendo a acurácia usada para escolher
    #                   o melhor modelo e a diferença usada se algum
    #                   empate ocorrer.
    # results_df: DataFrame do Pandas com todos resultados de todas os
    #             treinamentos e validações, utilizando a validação
    #             LOGO, para todos os grupos definidos pela LOGO e todos
    #             os modelos. As colunas dependem da validação, mas 
    #             existe uma coluna 'Model' que identifica o modelo, 
    #             usando o seu nome interno.
    # mean_scores_models_df: DataFrame com as pontuações médias das
    #                        validações cruzadas para cada um dos 
    #                        modelos avaliados, sendo as colunas as
    #                        pontuações e o índice das linhas os nomes
    #                        internos do modelo. As linhas do DataFrame
    #                        são ordenadas em ordem descrescente de
    #                        acordo com as pontuações e, como escolhemos
    #                        as pontuações máximas, então os modelos
    #                        são ordenados, do melhor para o pior, sendo
    #                        o modelo da primeira linha o melhor modelo
    #                        cujo nome interno foi retornado em 
    #                        best_model_name.
    (
      best_model_name, 
      best_model_score, 
      results_df, 
      mean_scores_models_df
    ) = discover_best_model.best_model(clean_dataset, suggestion_variables,
                                       application_variables, group_variables,
                                       suggestion_prediction_variable, 
                                       initialized_models)

    # Se a verbosidade estiver habilitada, imprime os DataFrames 
    # results_df e mean_scores_models_df. 
    if verbose:					
      # Imprimei o DataFrame results_df com as informações de todos os
      # treinamentos feitos em todas as validações cruzadas para todos
      # os modelos.
      print(f"➖➖➡️ Dataframe com os resultados das avaliações dos modelos:")
      print("\n", results_df.to_markdown(tablefmt="grid", floatfmt=".2f"), 
            "\n", sep="")

      # Imprime o DataFrame mean_scores_models_df com as pontuações 
      # médias dos modelos, que foram usadas para escolher o melhor 
      # modelo, que será o da primeira linha do DataFrame. Nos nossos
      # testes, as pontuações são a acurária (coluna 'accuracy') e o
      # negativo da diferença (coluna 'difference').
      print("➖➖➡️ Dataframe com os resultados médios para cada modelo, "
            "ordenado do melhor para o pior modelo:")
      print("\n", mean_scores_models_df.to_markdown(tablefmt="grid", 
                                                    floatfmt=".2f"), "\n", 
            sep="")

    # Retorna o nome interno do melhor modelo e o dicionário com as
    # pontuações desse melhor modelo.
    return best_model_name, best_model_score 

  # Foi passado um tipo inválido em um dos parâmetros da função
  # best_model.
  except ValueError as e:
    print("❌ Erro ao otimizar escolher o melhor modelo, um valor errado foi "
          "usado!")
    print(f"❌ Parâmetros do erro: {e.args}!")
    print(f"❌ Mensagem de erro: {str(e)}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como não foi possível otimizar os hiperparâmetros de algum dos
    # models, retorna None para os dois elementos da tupla.
    return None, None
  
  # Em um dos parâmetros de variáveis da função best_model, foi passada
  # uma variável que não existe no conjunto de dados da aplicação.
  except KeyError as e:
    print("❌ Erro ao escolher o melhor modelo, uma chave errada foi usada!")
    print(f"❌ Parâmetros do erro: {e.args}!")
    print(f"❌ Mensagem de erro: {str(e)}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como não foi possível otimizar os hiperparâmetros de algum dos
    # models, retorna None para os dois elementos da tupla.
    return None, None

  # Ocorreu alguma outra exceção inesperada.
  except Exception as e:
    print("❌ Erro desconhecido ao escolher o melhor modelo!")
    print(f"❌ Mensagem de erro: {str(e)}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como não foi possível escolher o melhor modelo, retorna None para
    # os dois elementos da tupla.
    return None, None

def generate_application_predictor(clean_dataset, best_model_name, 
                                   best_model_score, best_model_info,
                                   best_model_params, suggestion_variables, 
                                   application_variables, group_variables, 
                                   prediction_variables, verbose=False):
  """ 
  Função para gerar o preditor descrito no artigo do SSCAD. 

  Parâmetros:
    clean_dataset (DataFrame): Conjunto de dados já filtrado com os
                               testes para a aplicação para a qual 
                               desejamos escolher o melhor modelo dentre
                               os modelos que serão avaliados, que será
                               usado em todos os treinamendos da
                               validação cruzada, usando o LOGO, feita
                               quando as pontuações para cada um dos
                               modelos é calculada, para depois escolher
                               o modelo com a maior pontuação.
    best_model_name (str): Nome interno do melhor modelo, que teve a
                           maior pontuação dentre todos os modelos 
                           avaliados pela função choose_best_model, e 
                           que foi retornado por essa função.
    best_model_score (dict): Dicionário com as melhores pontuações do
                             melhor modelo, em que cada chave é uma das
                             pontuações usadas e o valor da chave é o
                             valor dessa pontuação. É somente usado para
                             na função para uma impressão se a
                             verbosidade estiver habilitada.
    best_model_info (dict): Dicionário com as informações do melhor
                            modelo. É o dicionário referenciado pela
                            chave do modelo no dicionário 'models' do
                            dicionário com as configurções de
                            treinamento.
    best_model_params (dict): Dicionário com os hiperparâmetros que
                              foram otimizados do melhor modelo e os 
                              hiperparâmetos fixos desse modelo. Cada
                              chave é um hiperparâmetro (fixo ou 
                              otimizado), e a referência da chave é o
                              valor do hiperparâmetro. Este dicionário
                              foi retornado pela função 
                              optimize_hyperparameters ao otimizar os
                              hiperparâmetros de cada modelo.                                          
    suggestion_variables (list[str]): Lista com os nomes das variáveis
                                      que definem as componentes de uma
                                      sugestão de configuração.
    application_variables (list[str]): Lista com os nomes das variáveis
                                       da aplicação, mas mão exatamente
                                       as originais passadas pelos 
                                       usuários, mas as que foram 
                                       convertidas, sendo as originais
                                       usadas somente se nenhuma
                                       conversão foi efetivamente
                                       feita (a conversão "copy" no
                                       script de otimização). Por 
                                       exemplo, no RAxML, não foi usado 
                                       o nome do arquivo, mas o tamanho
                                       desse arquivo.
    group_variavbles (list[str]): Lista com as variáveis, em geral as
                                  variáveis da aplicação definidas pelo
                                  usuário ou convertidas, que definem os
                                  grupos usados pela validação LOGO.
    prediction_variables (dict): Dicionário, obtido a partir da chave
                                 'estimated_parameters' do dicionário
                                 com as configurações da aplicação, que
                                 possui duas chaves, uma obrigatória,
                                 a 'suggestion', que define a variável
                                 alvo do modelo auxiliar usado pelo
                                 script de otimização para escolher a
                                 melhor sugestão de configuração (nos
                                 nossos testes, a EDP), e uma chave
                                 opcional, a 'time', que define a
                                 variável alvo do modelo a ser treinado
                                 se a chave for definida, que será 
                                 usado para estimar o tempo de execução
                                 da aplicação com a melhor sugestão de
                                 configuração.
    verbose (bool): Indica se a verbosidade está habilitada, ou seja, se
                    deverão ser impressas as mensagens de log.

  Retorna:
    SuggestionsPredictor: Se não ocorrer nenhum erro, refeência para o
                          objeto da classe SuggestionsPredictor com o
                          modelo auxiliar treinado usado para escolher
                          as melhores sugestões de configuração e, se
                          for definido, o modelo treinado para predizer
                          o tempo de execução da aplicação ao usar 
                          essa sugestão. O objeto também será usado para
                          obter outras informações, como as importâncias
                          das variáveis após o treinamento do modelo
                          auxilar usado para escolher a melhor sugestão,
                          o DataFrame com os oráculos para cada
                          combinação dos parâmetros das variáveis de
                          configuração e da aplicação, e a
                          funcionalidade usada pelo script de
                          otimização para escolher a melhor sugestçao de
                          configuração. Este é efetivamente o Preditor
                          descrito no artigo do SSCAD. Caso algum erro
                          ocorra ao gerar o preditor, retorna None.
  """

  # Se a verbosidade estiver habilitada, imprime as informações sobre
  # a construção do preditor, no caso, o estimador
  try:
    # Imprime uma mensagem informando os dados do melhor modelo usado
    # ao criar o preditor.
    print("➖➡️ Treinando agora o preditor com o melhor estimador "
          f"{best_model_name} (pontuação: {best_model_score}), usando "
          f"{best_model_params} como os hiperparâmetros customizados.")

    # Mostra uma mensagem informando qual variável alvo será predita
    # pelo modelo auxiliar cujas predições serão usads para escolher a
    # melhor sugestão de configuração.
    print("➖➡️ O modelo auxiliar usado ao escolher a melhor sugestão de "
          "configurção usará a variável alvo "
          f"{prediction_variables['suggestion']}.")

    # Verifica se o modelo para predizer o tempo estimado de execução da
    # aplicação quando executaar com as alocações definidas de acordo
    # com a melhor sugestão de configuração.
    if 'time' in prediction_variables.keys():
      print("➖➡️ O modelo usado para estimar o tempo de execução da "
            "aplicação quando executar com a melhor sugestão de configuração "
            f"usará a variável alvo {prediction_variables['time']}.")

    # Descobre e importa o modelo com o nome interno best_model_name, 
    # de modo dinâmico, sendo que o  caminho para importar esse modelo é 
    # definido na chave 'import_path' do dicionário best_model_info. 
    # Como o caminho indica como o modelo seria acessado diretamente, 
    # importando somente a bibliotea básica do Python que está 
    # implementado o modelo, por exemplo, para o regressor Extra Trees
    # do scikit-learn 'sklearn.ensemble.ExtraTreesRegressor', então 
    # usamos o rsplilt da classe string, para separar as strings por um 
    # separador, no caso o ",", a partir da direita (o "r" em rsplit), 
    # em duas strings de acordo com o último ponto, sendo o primeiro,
    # armazenado em module_path, o caminho que seria usado em um "from" 
    # para importar um objeto específico da biblioteca (ou módulo), e o
    # segundo, armazenado em model_name, o nome da classe do objeto. No 
    # caso das Extra Trees, module_path seria uma referência para a 
    # string "sklearn.ensemble" e model_neme seria uma referência para a
    # string "ExtraTreesRegressor".
    module_path, model_name = best_model_info['import_path'].rsplit(".", 1)

    # Uma vez que definimos o caminho completo na biblioteca ou 
    # módulo, usamos a função import_module da biblioteca padrão do
    # Python importlib para dinâmicamente importar a biblioteca que
    # implementa o modelo, sendo a referência ao objeto que representa
    # a biblioteca armazenada em model_module.
    model_module = importlib.import_module(module_path)

    # Uma vez que a biblioteca tenha sido importada em um objeto
    # referenciado por model_module, os seus atributos serão o que
    # podemos importar dessa bibliotca, ou seja, um desses atributos
    # deveria ser o nome daa classe do objeto que implementa o modelo
    # para o qual iremos otimizar os seus hiperparâmetros, supondo que
    # o nome passado na chave 'import_path' do dicionário model_info
    # está correto. Se estiver correto, retorna a referência para a
    # classe do modelo e, em caso contrário (se o atributo não 
    # existir, retorna None).
    model = getattr(model_module, model_name, None)

    # Verifica o objeto model_module tem o atributo model_name.
    if model is not None:
      # Como o objeto model mão é None, ou seja, o objeto model_module 
      # tem o atributo model_name,  então vamos criar um objeto da 
      # classe SuggestionsPredictor, usada para construir o preditor da
      # aplicação, treinando com todos os dados do conjunto de dados
      # clean_dataset o modelo auxiliar usado ao escolher a menor
      # sugestão de configuração e, se definido, o modelo para estimar
      # o tempo de execução quando a aplicação é executada com a melhor
      # sugestão de configuração, e armazenar a rederência a esse objeto
      # em predictor.
      predictor = SuggestionsPredictor()

      # Executa a função fit do objeto preditor para treimar o modelo
      # auxiliar com todos os testes do conjunto de dados, usando as
      # variáveis de configuração, dadas em suggestion_variables e as
      # varipaveis da aplicação, dadad em application_variables, usando
      # a variável alvo definida pela chave 'sugeestion' do dicionário
      # prediction_variables. Se existir a chave 'time' no dicionário
      # prediction_variables, o modelo usado para estimar o tempo de
      # execução da aplicação quando utilizar a melhor sugestão de
      # configuração, é treinado também com todos os testes em 
      # clean_dataset, utilizando as variáveis dadas em 
      # suggestion_variables e application_variables, e a variável alvo
      # definida pela chave 'time'. AS variáeis dadas em group_variables
      # são somente usadas para retornar o DataFrame com os oráculos
      # para o modelo auxiliar usado para escoher a melhor sugestão de
      # configuração. O retorno é descartado porque, seguindo o padrão
      # dos métodos fit da scikit-learn, é uma referência para o próprio
      # objeto, ou seja, a mesma referência armazenada em predictor. O
      # último parâmetro, a constande DEBUG_CODE, habilita informações
      # adiconais de depuração, para auxiliar a correção de eventuais
      # erros (DEBUG_CODE é habilitado usando a variável de ambiente
      # APPOPTIMIZER_DEBUG definida com true, T, 1 ou on, sendo que as
      # strings não são sensíveis ao caso).
      predictor.fit(clean_dataset, suggestion_variables, application_variables,
                    group_variables, prediction_variables, model, 
                    best_model_params, DEBUG_CODE)

      # Se a verbosidade estiver habilidata, imprime as informações 
      # referentes ao modelo auxiliar usado para escolher a melhor 
      # sugestão de configuração.
      # TODO; Devemos fazer algo similar para o modelo que prediz o
      #       tempo de execução da aplicação ao usar a melhor sugestão
      #       de configuração?
      if verbose:
        # Imprime o DataFrame com os oráculos para o modelo auxilar, 
        # usando todas as combinações dos possíveis valores da lista de
        # variáveis dadas em group_variables. Como antes, a constante
        # DEBUG_CODE gera informações adicionais sobre  geração dos
        # oráculos para a depuração;
        print('➖➖➡️ Dataframe do oráculo:')
        oracle_df = predictor.get_oracle(verbose=DEBUG_CODE)
        print("\n", oracle_df.to_markdown(tablefmt="grid", floatfmt=".2f"), 
              "\n", sep="")

      # Tenda obter as importâncias das variáveis de entrada ao treinar
      # o modelo auxiliar, ou seja, as variáveis dadas em 
      # suggestion_variables e application_variables. Novamente, a 
      # constante DEBUG_CODE mostra informações adicionais sobre a
      # geração das importâncias que podem ser usadas para depuração.
      importances_df = predictor.get_importances(verbose=DEBUG_CODE)

      # Verifica se a verbosidade está habilitada e se as improtâncias
      # foram definidas.
      if verbose and importances_df is not None:
        # Se o estimador usado ao treinar o modelo auxiliar definir as 
        # importâncias para as variáveis de entrada, mostra o DataFrame
        # com essas importâncias. 
        print('➖➖➡️ Dataframe com as importâncias do modelo:')
        print("\n", importances_df.to_markdown(tablefmt="grid", 
                                               floatfmt=".2f"), 
              "\n", sep="")
      elif verbose:
        # Se as importâncias não são definidas pelo estimador usado ao
        # treinar o modelo auxiliar, informa isso em uma mensagem se a
        # verbosidade estiver habilitada.
        print("➖➖➡️ O modelo não avalia as importâncias das "
              "características.")

      # Como não ocorreram erros ao treinar o modelo auxiliar usado para
      # escolher a melhor sugestão de configuração e, se for definido,
      # o modelo para estimar o tempo de execução da aplicação ao ser
      # executada com essa configuração, retorna uma referência ao
      # objeto do preditor.
      return predictor
    else:
      # Como não foi encontrado uma classe com o nome model_name na
      # bibloreca referenciada por model_module, mostra um erro
      # informando que o modelo não existe e, por coerência com o erro
      # do módulo não existir, geramos a exceção ImportError.
      print(f"❌ O caminho de importação {best_model_info['import_path']} é "
            f"inválido. A classe do modelo {model_name} não existe na "
            f"bibloteca referenciada por {module_path}!")
      raise ImportError(f"O modelo {model_name} não existe no módulo "
                        f"{module_path}!", name=module_path)
    
  # Foi passado um tipo inválido em um dos parâmetros da função
  # best_model.
  except ValueError as e:
    print("❌ Erro ao otimizar escolher o melhor modelo, um valor errado foi "
          "usado!")
    print(f"❌ Parâmetros do erro: {e.args}!")
    print(f"❌ Mensagem de erro: {str(e)}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como não foi possível otimizar os hiperparâmetros de algum dos
    # models, retorna None para os dois elementos da tupla.
    return None
  
  # Em um dos parâmetros de variáveis da função best_model, foi passada
  # uma variável que não existe no conjunto de dados da aplicação.
  except KeyError as e:
    print("❌ Erro ao escolher o melhor modelo, uma chave errada foi usada!")
    print(f"❌ Parâmetros do erro: {e.args}!")
    print(f"❌ Mensagem de erro: {str(e)}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
    return None

  # Ocorreu um erro ao importar um dos modelos.
  except ImportError as e:
    print("❌ Erro ao gerar o preditor para a aplicação, uma classe não "
          f"existe no módulo {e.name}!")
    print(f"❌ Mensagem de erro: {str(e)}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como não foi possível otimizar os hiperparâmetros de algum dos
    # models, retorna None para os dois elementos da tupla.
    return None

  # Ocorreu um erro ao importar um dos modelos.
  except ModuleNotFoundError as e:
    print("❌ Erro ao gerar o preditor para a aplicação, uma biblioteca "
          f"{e.name} errada foi usada")
    print(f"❌ Mensagem de erro: {str(e)}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como não foi possível otimizar os hiperparâmetros de algum dos
    # models, retorna None para os dois elementos da tupla.
    return None

  # Ocorreu alguma outra exceção inesperada.
  except Exception as e:
    print("❌ Erro desconhecido ao escolher o melhor modelo!")
    print(f"❌ Mensagem de erro: {str(e)}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")

    # Como não foi possível escolher o melhor modelo, retorna None para
    # os dois elementos da tupla.
    return None
  
def save_application_predictor(best_model_acronym_name, best_model_name, 
                               predictor, predictors_file_path, 
                               predictors_info_config,
                               predictors_info_config_obj, 
                               suggestion_prediction_variable, 
                               application_name, verbose=False):
  """ 
  Função para salvar o preditor da aplicação, com o modelo auxiliar 
  treinado, com o conjunto de dados completo e filtrado, para predizer
  a variável definida pela chave 'suggestion' do dicionário referenciado
  pela chave 'estimated_parameters' do dicionário com os arquivos de
  comfiguração dessa aplicação e, se for treinado, o modelo para
  predizer o tempo de execução da aplicação ao ser executada usando a
  melhor sugestão de configuraçãoa, que prediz a variável alvo dada pela
  chave 'time' do dicionário referenciado pela chave
  'estimated_parameters'. Esse último modelo somente será treinado se
  a chave 'time', que é opcional, for definida.

  Parâmetros:
    best_model_acronym_name (str): Acronym do melhor modelo, que teve
                                   a maior pontuação dentre todos os
                                   modelos avaliados pela função 
                                   choose_best_model, e que foi 
                                   retornado por essa função.
    best_model_name (str): Nome do melhor modelo definido pela chave 
                           'name' no arquivo de configuração do melhor
                           modelo com o nome interno 
                           best_model_internal_name. O nome é usado para
                           compor o nome do arquivo com o preditor salvo. 
    predictor (SuggestionsPredictor): Referência para o objeto com o
                                      preditor da aplicação, gerado pela
                                      função 
                                      generate_application_predictor, 
                                      com o modelo treinado para
                                      auxiliar, pelo script de 
                                      otimização, a escolha da melhor
                                      sugestão de configuração e, se
                                      foi treinado, o modelo para
                                      estimar o tempo de execução da
                                      aplicação usando essa sugestão. 
                                      Como observamos, este objeto é a
                                      implementação do Preditor descrito
                                      no artigo do SSCAD.
    predictors_file_path (str): Caminho completo para o diretório em que
                                são armazenados os preditores das
                                aplicações, no formato .pickle em que
                                os objetos do python são passados por
                                um processo chamado de serialização para
                                poder armazená-los e usá-los mais tarde,
                                no caso, pelo script de otimiziação, e
                                o arquivo de configuração dos 
                                preditores.
    predictors_info_config (dict): Referência para o dicionário com o
                                   arquivo de configuração dos
                                   preditores, sendo que cada chave é o
                                   nome interno da aplicação, e o seu
                                   valor é o nome do arquivo com o 
                                   preditor salvo, que será lido e usado
                                   pelo script de otimização. A 
                                   configuração será atualizada com uma
                                   chave para a aplicação, ou a 
                                   alteração do nome da chave se ela já
                                   existir, referenciando uma string com
                                   o nome do arquivo em que o preditor
                                   foi salvo.
    predictors_info_config_obj (PredictorsInfoConfig): Referência para o
                                                       objeto do 
                                                       preditor, 
                                                       necessária para
                                                       salvar o arquivo
                                                       de configuração
                                                       atualizado com o
                                                       novo preditor
                                                       treinado para a
                                                       aplicação.
    suggestion_prediction_variable (str): Nome da variável alvo usada ao
                                          treinar o modelo auxiliar
                                          usado para escolher a melhor
                                          sugestão de configuração. O
                                          nome é usado para compor o
                                          nome do arquivo com o preditor
                                          salvo.
    application_name (str): Nome interno da aplicação, também usado para 
                            compor o nome do arquivo com o preditor 
                            salvo. 
    verbose (bool): Indica se a verbosidade está habilitada, ou seja, se
                    deverão ser impressas as mensagens de log.

  Retorna:
    bool: True se o preditor foi salvo com sucesso e o arquivo de 
          configuração dos preditores atualizado com sucesso e False se
          ocorreu algum erro ao salvar o preditor ou o arquivo de 
          configuração dos preditores.
  """

  # Define o caminho completo do nome do arquivo do preditor da 
  # aplicação que teve os seus modelos treinados pela função 
  # generate_application_predictor. O caminho é composto por 
  # predictors_file_path, o caminho para o diretório com os preditores
  # e com o arquivo de configuração do preditores, e o nome do preditor,
  # que é composto pelo nome interno da aplicação para o qual o prediror
  # foi gerado, dado em application_name, o nome do melhor estimador que
  # foi usado para treinar o modelo auxiliar e, se definido, o modelo
  # para predizer os tempos armazenados no preditor, dado em 
  # best_model_name, a variável alvo do modelo auxiliar do preditor, 
  # dada em suggestion_prediction_variable, e a extensão .pickle, pois
  # o preditor é armazenado neste formato.
  preditor_file_name = (predictors_file_path / 
                        f"{application_name}_{best_model_name}_"
                        f"{suggestion_prediction_variable}.pickle")
  
  # Imprime uma mensagem informando que o preditor gerado para a
  # aplicação será salvo no arquivo cujo caminho está definido pelo
  # objeto Path com a referência armazenada em preditor_file_name,
  # mostrando o nome da aplicação (dado em application_name), o nome 
  # interno do melhor modelo (dado em best_model_acronym_name), o nome
  # do modelo definido pela chave 'name' do dicionário referenciado pelo
  # nome interno do melhorvmodelo (dado em best_model_name), e o caminho
  # em que o preditor será armazenado, d
  print("➖➡️ Salvando o preditor treinado para a aplicação "
        f"{application_name} como o modelo {best_model_acronym_name} (nome "
        f"{best_model_name}) no arquivo do caminho {preditor_file_name}.")

  # Define uma variável old_preditor_file_name com o nome do preditor
  # anteriormente criado para a aplicação, se isso foi feito antes, ou
  # seja, se o nome interno da aplicação application_name é uma chave 
  # válida do dicionário predictors_info_config. Como usamos a função
  # get do dicionário com o parâmetro default definido como None, 
  # old_preditor_file_name será o nome do preditor antigo se a chabe
  # application_name existir, ou None se a chave não existir, ou seja,
  # nenhum preditor foi criado antes para a aplicação. A variável será
  # usada para remover o preditor anterior da aplicação se ele existir
  # (a variável não for None)e o arquivo de configuração dos 
  # preditores tiver sido atualizado com sucesso.
  old_preditor_file_name = predictors_info_config.get(application_name, None)

  # Chama a função save_predictor do obejto da classe 
  # SuggestionsPredictor, que salva o preditor no arquivo definido pelo
  # caminho preditor_file_name mo formato .pickle. O formaro pickle do
  # Python faz o que se chama de a serialização de um objeto, no caso,
  # o objeto da classe SuggestionsPredictor, que permite que esse objeto
  # seja salvo em um arquivo e depois possa ser usado, quando for lido
  # e desserizlizado desserizlizado do arquivo .pickle salvo, como se
  # posse o objeto salvo. O processo de desserizlizado será feito no
  # script de otimização quando o objeto for usado para escolher a 
  # melhor sugestão de configuração usando a função get_suggestion do
  # objeto. também existe uma função para obter várias sugestões ao
  # mesmo tempo, get_suggestions, mas essa função ainda não foi usada,
  # somente testada. Se o arquivo foi salvado corretamente, status será
  # True e se algum erro ocorreu e o arquivo não pôde ser salvo, status
  # será False.
  status = predictor.save_predictor(preditor_file_name)

  # Verifica se o arquivo foi salvo ou não com sucesso.
  if status:
    # Como o arquivo foi salvo com sucesso, pois status é True, então
    # primeiramenre, se a verbosidade estiver habilitada, vamos mostrar
    # uma mensagem informando que o arquivo de configuração vai ser
    # atualizado. A mensagem mostra o caminho completo do predotor, 
    # definido pelo objeto da classe Path cuja referência está 
    # armazenada em preditor_file_name e o nome da aplicação;
    if verbose:
      print("➖➡️ Salvando a informação do caminho do preditor "
            f"{preditor_file_name} para a aplicação {application_name} no "
            "arquivo de configuração dos preditores.")
      
    # Atualiza a informação do preidtor para a aplicação com o nome
    # interno armazanado em application_name. Como as configurações dos
    # preditores são salvas em um dicionário cujas chaves são os nomes
    # inyternos dos arquivos e cuja referências são os nomes dos 
    # arquivos dos preditores sem os caminhos (como os caminhos de todos
    # os preditores são os mesmos, o diretório com todo os preditores e
    # o arquivo de configuração dos preditores, o caminho é
    # reconstruído pelo script de otimização), então o comando a seguir
    # irá ou criar uma nova chave no dicionário referenciado por 
    # predictors_info_config, se a chave application_name não existir no
    # dicionário, ou irá atualizar a referência dela para o objeto str
    # referenciado por preditor_file_name.name, sendo name um atributo
    # do objeto da classe Path que define o nome do arquivo do caminho
    # armazenado no objeto a atring cuja referência será associada à 
    # chave application_name.
    predictors_info_config[application_name] = preditor_file_name.name

    # Usa a função save_predictors_info_config do objeto da classe
    # PredictorsInfoConfig, que leu o arquivo de configuração dos
    # preditores no dicionário referenciado por predictors_info_config,
    # com o dicionário alterado e passado no parâmetro
    # predictors_info_config. O nome do arquivo será o mesmo usado ao
    # ler o arquivo de configuração dos preditores na função 
    # train_commanda descrita a seguir, que gerencia todos os passos de
    # geração do preditor para cada uma das aplicações passadas no
    # comando train. A função retorna True se foi possível atualizar o
    # arquivo de configuração dos preditores ou criá-lo se ele ainda
    # não existia, ou False se não foi possível salvar o arquivo.
    status = predictors_info_config_obj.save_predictors_info_config(
      predictors_info_config
    )

    # Verifica se o arquivo de configuração dos preditores foi salvo com
    # sucesso.
    if status:
      # Se a verbosidade está habilitada, informa que o arquivo de 
      # configuração dos preditores foi atualizado, sendo a atualização
      # ocorrida para a aplicação application_name
      if verbose:
        print("➖➡️ Informação do caminho salva com sucesso para a aplicação "
              f"{application_name}: a chave {application_name} do dicionário "
              "de configuração dos preditores foi alterada para "
              f"{preditor_file_name.name}")

      # Verifica se anteriormente foi treinado um preditor para
      # aplicação, o que ocorrerá somente se a variável
      # old_preditor_file_name tiver um nome válido, armazenapd em uma
      # string (objeto str) e não None.
      if old_preditor_file_name is not None:
        # Como existe um preditor anteriormente criado para a aplicação,
        # vamos primeiramente descobrir o caminho. O caminho será
        # composto pelo caminho definifo pelo objeto da classe Path
        # predictors_file_path, que armazena o caminho do diretório com
        # os preditores e o arquivo de configuração da aplicação, e
        # o nome do preditor dado em old_preditor_file_name.
        old_preditor_file_path = (predictors_file_path / 
                                  f"{old_preditor_file_name}")

        # Se a verbosidade estiver habilitada, mostra uma mensagem
        # informando que o preditor criado anteriormente para aplicação
        # será removido porque não é mais necessário.
        if verbose:
          print(f"➖➡️ Removendo o preditor antigo {old_preditor_file_name} "
                f"da aplicação {application_name} porque ele não é mais "
                "necessário!")  

        # Usa a função unlink do objeto da classe Path
        # old_preditor_file_path, com o caminho completo do preditor
        # criado anteriormente, para remover esse arquivo do preditor,
        # sem gerar erros se por acado o arquivo não existir ou não
        # puder ser removido, devido ao parâmetro nomeado missing_ok ter
        # sido definido como True.
        # TODO: Será que deveríamos deixar o erro ocorrer e capturar em
        #       uma exceção?
        old_preditor_file_path.unlink(missing_ok=True)
    else:
      # Se ocorreu um erro ao atualizar o arquivo de configuração da
      # aplicação, então primeiramente mostramos uma mensagem informando
      # o erro ao uuário do script de treinamento.
      print("❌ Erro ao salvar o arquivo de configuração dos preditores "
            f"atualizado para a aplicação {application_name}!")

      # Mostra uma mensagem informando que o preditor criado será
      # removido porque não poderá ser  usado pelo script de otimização.
      print(f"❌ Removendo o preditor criado {preditor_file_name}, já que não "
            "foi possível atualizar o arquivo de configuração dos preditores!")  
      
      # Como não foi possível atualizar o arquivo de configuração dos
      # preditores, o arquivo com o novo preditor criado deverá ser
      # removido, pois o arquivo de configuração dos preditores não terá
      # uma refeência para este arquivo e, com isso, ele não poderá ser
      # usado pelo script de otimização. Logo, vamos novamente usar a
      # função unlink do objeto da classe Path preditor_file_name, com o 
      # caminho completo do preditor que foi criado, para remover esse
      # arquivo do preditor, sem gerar erros se por acado o arquivo não
      # puder ser removido, devido ao parâmetro nomeado missing_ok ter
      # sido definido como True.
      # TODO: Se não for possível atualizar o arquivo de configuração dos
      #       preditores com o novo preditor, será que devemos remover o
      #       preditor criado?
      preditor_file_name.unlink(missing_ok=True)      
  else:
    # Como ocorreu um erro ao salvr o arquivo serializado com o preditor
    # criado, mostramos uma mensagem informando isso ao usuário do
    # script de treinamento.
    print(f"❌ Erro ao salvar o preditor no caminho {preditor_file_name} "
          f"para a aplicação {application_name}!")

  # Retorna status que será. como observamos anteriormente, True se o
  # preditor criado para a aplicação foi salvo com sucesso e também o
  # arquivo de configuração dos preditores foi salvo com sucesso, ou 
  # False em caso contrário, ou seja, se ocorreu um erro ao salvar o
  # arquivo do preditor ou o arquivo com a configuração dos preditores.
  return status

def train_command(applications_name, applications_config, training_config, 
                  system_config,verbose=False):
  """
  Função para executar o comando train, que gera o preditor, que será
  usado pelo script de otimização, para cada aplicação passada como
  parâmtro, sendo a aplicação identificada pelo seu nome interno, ou
  seja, o nome dado pela chave 'name' do arquivo de configuração da
  aplicação e do dicionário com as configuraçẽos da aplicação obtido a
  partir deste arquivo. Cada aplicação listada tem o seu preditor
  idependentemente gerado das outras aplicações. Para cada aplicação em
  que o preditor foi gerado com sucesso, a sua entrada no arquivo de
  configuração dos preditores ou será alterada para o nome do arquivo do
  novo preditor, se a aplicação já tinha um preditor criado, sendo que
  o preditor anterior da aplicação será removido, ou será criada uma
  nova chave para a aplicação referenciando a string com o nome do 
  preditor treinado para a aplicação.

  Parâmetros:
    applications_name (list[str]): Lista com os nomes internoa das 
                                   aplicações para as quais desejamos
                                   construir um novo preditor.
    applications_config (dict): Dicionário com todas as configurações
                                das aplicações que podem ser otimizadas
                                pelo script de otimização, sendo que
                                cada chave, associada e identificando
                                uma aplicação, tem uma referência para o
                                dicionário com as configurações dessa
                                aplicação.
    training_config (dict): Dicionário com todas as configurações
                            de treinamento. Como os dicionários com as
                            configurações de treinamento dos modelos são 
                            armazenados no diiconário cuja referência é
                            definida pela chave "models", então os nomes
                            dos modelos que podem ser treinados são as
                            chaves desse dicionário, sendo que cada
                            chave referencia um diconário para o modelo
                            com o nome interno da chave com todas as
                            configurações necessárias para otimizar os
                            hiperparâmetros do modelo, os 
                            hiperparâmetros com valores fixos do modelo,
                            o nome do modelo, e o nome usado para
                            dinamicamente criar um objeto do modelo.
    system_config (dict): Dicionário com as configurações gerais
                          relacionadas aos scripts de otrimização e de
                          treinamento. A função irá passar o dicionário
                          para outras funções, que usarão os campos
                          destritos no cabeçalho das funções,
    verbose (bool): Indica se a verbosidade está habilitada, ou seja, se
                    deverão ser impressas as mensagens de log.

  Retorna:
    bool: True se todos os preditores de todas as aplicações foram 
          criados e disponibilizados com sucesso, ou seja, se cada
          preditor foi criado, salvo com sucesso, em um arquivo no
          formato .pickle, e a entrada da aplicação no arquivo de
          configuração dos preditores foi alterado e o arquivo
          atualizado salvo com sucesso.
  """

  try:
    # Define o nome da aplicação para None, para não mostrar um nome da
    # aplicação se uma exceção, devido a algun erro de programação no
    # código, ocorrer antes de começarmos a treinar as aplicações.
    application_name = None

    # Inicializa a variável booleana status, que indica se algum erro
    # ocorreu em um dos passos ao gerar o preditor para uma aplicação.
    # Como inicialmente não temos erros, então o valor inidial dela é
    # True
    status = True

    # Primeiramente vamos ler o arquivo de configuração dos preditores,
    # pois saber se é possível ler o arquivo é importante pois
    # precisamos ler o arquivo para remover os preditores antigos e
    # salvar o arquivo para armazenar os nomes dos novos preditores para
    # aplicações já treinadas, ou para novas aplicações, A Função
    # retorna os seguintes parâmetros:
    #
    # predictors_file_path: Objeto da classe Path com o caminho do 
    #                       diretório com o arquivo de configruação dos
    #                       preditores e os arquivos dos preditores.
    # predictors_info_config: Dicionário com o arquivo de configuração
    #                         dos preditores lido, se nenhum erro
    #                         ocorreu ao ler o arquivo, sendo cada chave
    #                         o nome da aplicação e o valor da chave o
    #                         nome do arquivo com o preditor da
    #                         aplicação. Se não foi possível ler o
    #                         arquivo, o valor será None.
    # predictors_info_config_obj: Objeto da classe PredictorsInfoConfig
    #                             que foi criado para ler o arquivo de
    #                             configuração dos preditores. O objeto
    #                             será necessário para atualizar o 
    #                             arquivo de configuração com o nome do
    #                             arquivo do nome preditor criado para
    #                             cada aplicação passada no comando
    #                             train.
    (
      predictors_file_path, 
      predictors_info_config, 
      predictors_info_config_obj
    ) = read_predictor_info_config(system_config, verbose=verbose)

    # Verifica se foi possível ler o arquovo de configuração dos
    # preditores.
    if predictors_info_config is None:
      # Caso predictors_info_config seja None, então ocorreu um erro ao
      # ler o arquivo de configuração. Logo, geremoas uma mensagem 
      # informando sobre o erro.
      print("❌ Não foi possível ler o arquivo de configuração com as "
            "informações dos preditores, logo não foi possível treinar os "
            "modelos para as aplicações!")

      # Como não foi possível ler o arquivo de configuração dos
      # preditores, então não será possível criar nenhum preditor para
      # nenhuma aplicação e, com isso, retornamos False.
      return False

    # Vamos começar a criação dos preditores para cada aplicação dada
    # na lista applications_name.
    for application_name in applications_name:
      # Vamos começar a fazer os passos necessários para construir o 
      # preditor para a aplicação application_name. Primeiramente, 
      # verificamos se podemos criar um preditor para a aplicação; Isso
      # somente será possível se a aplicação possuir um arquivo de 
      # configuração da aplicação, o que ocorrerá somente se existir,
      # nas cjhaves do dicionário applications_config, uma chave com o
      # nome da aplicação, porque cada aplicação que possuía um arquivo
      # teve uma chave criada, com o nome da aplicação definido pela
      # chave 'name' do arquivo de configuração desta apicação.
      if application_name in applications_config.keys():
        # Se existir a chave com o noem da aplicação application_name em
        # applications_config, ebtão podemos criar um preditor para essa
        # aplicação. Para facilitar o acesso, vamoe definir a varável 
        # application_info com o dicionário com as configurações da 
        # aplicação, incluindo as configurações de treinamento, sendo o
        # dicionário referenciado pela chave application_iname.
        application_info = applications_config[application_name]

        # Se a verbosidade estiver habilitada, mostra uma mensagem 
        # informando que vamos começar o processo de geração do 
        # preditor para a aplicação.
        if verbose:
          print("\n➡️ Treinando todos os modelos para a aplicação "
                f"{application_name}")	

        # Para faciliar o acesso, definimos a variável
        # suggestion_variables com os nomes das varíaveis que compõe uma 
        # configuração, como o número de nós, processos por nó e threads
        # por processo.
        suggestion_variables = application_info['suggestions_parameters'] 

        # Para faciliar o acesso, definimos a variável a
        # application_variables com os nomes das varíaveis que definem
        # os parâmetros da aplicação que serão usados nos treinamentos
        # de todos os modelos, tanto os finais como so intermediários
        # das fases de otiização dos hiperparâmetros e escolha do
        # melhor modelo.
        application_variables = application_info['application_parameters']  

        # Para faciliar o acesso, definimos a variável group_variables
        # com os nomes das varíaveis usadas para construir os gruoos
        # usados pela validação LOGO nas fases de otimização dos
        # hiperparâmetros e escolha do melhor modelo. Os grupos serão
        # obtidos por todas as combinações de valores diferentes deasas
        # variáveis no conjunto de dados da aplicação, depois de
        # filtrado.
        group_variables = application_info['training']['group_parameters'] 

        # Para faciliar o acesso, definimos a variável filter_variables
        # com os nomes das varíaveis que serão usadas para fazer a
        # filtragem do conjunto de dados com o objetivo de remover os
        # outliers que podem comprometer o treinamento dos modelos
        # gerados em todo o processo de criação do preditor. Em geral,
        # são as variáveis alvo e/ou variáveis relacionadas à elas. Nos
        # nosso s testes, usamos as varáveis "ElapsedRaw" (tempo de
        # execução, obtido pelo saccct), "Consumo de Energia Total (J)"
        # consumo de energia, medido nos testes utilizando a arquitetura
        # RAPL da Intel) e "EDP" (o produto de "ElapsedRaw" por 
        # "Consumo de Energia Total (J)""). Apesar de termos usados as
        # variáveis em todas as aplicações oara as quais criamos um 
        # preditor, as variáveis podem variar dependendo da aplicação, 
        # pois são definidas no arquivo de configurção da aplicação.
        filter_variables = application_info['training']['filter_parameters']

        # Para faciliar o acesso, definimos a variável
        # prediction_variables com o dicionário em que a chave
        # obrigatória 'suggestion' define a variável alvo dos modelos
        # usados ṕara otimizar os hiperparâmetros e escolher o melhor
        # modelo e para o treinamento do modelo auxiliar, associado ao
        # preditor criado, usado pelo script de otimização para escolher
        # a melhor sugestão de configuração.
        prediction_variables = application_info['estimated_parameters']        

        # Para faciliar o acesso, definimos a variável 
        # suggestion_prediction_variable com o nome da variável alvo 
        # obrigatoria definda, como objervamos anteriomente, pela
        # chave 'suggestion' do dicionário prediction_variables. 
        suggestion_prediction_variable = prediction_variables['suggestion']

        # Define as variáveis de enrrada dos modelos e das validações,
        # usadas nos treinamentos dos modelos e nas construções dos
        # grupos da validação LOGO. Como uma variável pode estar em mais
        # de uma das variáveis definidas anteriormente, por exemplo, as
        # variáveis de aplicação BootStrap e Tamanho que usamos nos
        # treinamentos do RAxML estão tando em application_variables,
        # como em group_variables. Para remover as duplicatas, 
        # convertemos a concatenação das listas suggestion_variables,
        # application_variables e group_variables, que é uma nova lista,
        # em um conjunto do python, que automaticamente remove 
        # duplicatas, no caso, strings iguais, e depois convertemos o
        # conjunto sem as duplicatas novamente para uma lista.
        input_variables = list(
            set(
                suggestion_variables
                + application_variables
                + group_variables
            )
        )

        # Define todas as variáveis do conjunto de dados da aplicação 
        # usadas em todos os processamentos ao criar o preditor de cada
        # aplicação, que será a concatenação das listas input_variables,
        # filter_variables e os valores das chaves do dicionário 
        # prediction_variables (no momento temos no máximo duas, a
        # obrigatória 'suggestion' com a variável alvo usada no 
        # treinamento de todos os modelos da fase de otimização dos
        # hiperparâmetros e escolha do melhor modelo e no treinamento do
        # modelo auxiliar usado para escolher a melhor guestão de
        # configuração). Novamente usamos a conversão da lista
        # concatenada para conjunto e depois para a lista para remover
        # possíveis nomes iguais de variáveis. Como essa variável é
        # excluivamente usada pela função read_training_data para ler o
        # conjunto de dados da aplicaççao, a remoção das duplicatas é
        # para evitar que, no DataFrame que armazena o conjunto de
        # dados, existam colunas duplicadas.
        dataset_colunms = list(
            set(
                input_variables
                + filter_variables
                + list(prediction_variables.values())
            )
        )
    
        # Para faciliar o acesso, definimos a variável outlier_limit com
        # o valor em ponto flutuante que define como é defnido o
        # intervalo de exclusão dos outiliers, definido na chave 
        # 'outlier_limit' do dicionário com as informações de filtragem
        # (no momento, somente a outlier_limit) referenciado pela chave
        # 'filter' do diconário com as configurações de treinamento. O
        # intervalo será [mad-outlier_limit, mad+outlier_limit], onde
        # mad é o desvid mediano absoluto de todos os valores de uma das
        # variáveis de flitragem defindas em filter_variables. A
        # flitrage é feita separadamente para cada variável em
        # filter_variables.
        outlier_limit = training_config['filter']['outlier_limit']

        # Para faciliar o acesso, definimos a variável modelos com a
        # referência ao dicionário, definido pela chave 'models1 do 
        # arquivo de configuração de treinamento, com as configurações
        # de cada modelo a ser avaliado, sendo cada chave do dicionário
        # o modelo, e a referência desta chave o dicionário com as
        # configurações deste modelo, que incluem, como vimos, os
        # hiperparâmetros a serem otimizados, os hiperparâmetros fixos,
        # o nome do modelo, e o identificador usado para importar
        # dinamicamente a classe do modelo.
        models = training_config['models']

        # Para faciliar o acesso, definimos a variável dataset_list com
        # a lista dos arquivos que armazenam, no formato cvs, todos os
        # conjunto de dados que serão concetenados em um único conjunto 
        # de dados que será  posteriormente passado para a fase de
        # filtragem para a remoção dos outliers. Não é obrigstório
        # definir vários arquivos, ou seja, a lista pode ser composta
        # por somente um arquivo. Fiz a implementação com uma lista
        # pensando em dados futuros, que poderiam ser axenados à parte,
        # permitindo a preservação do conjunto de dados originalmente
        # usado nas fases prelinimares da construção do preditor da
        # aplicação.
        dataset_list = application_info['training']['dataset_files']

        # Chama a função read_training_data que lê todos os conjuntos
        # de dados cujos nomes de arquivo são definidos pela lista 
        # dataset_list, e retorna o conjunto de dados completo lido, 
        # obtido pela concatenação de todos os conjunto de dados, em um
        # DataFarame do Pandas, se nenhum erro de leitura ocorreu, ou
        # None se algum erro ocorreu ao ler os conjutos de dados e 
        # concatená-los. O retorno da função, ou seja o conjunto de
        # dados no caso de sucesso ou None em caso contrário, é
        # armazenado na variável dataset.
        dataset = read_training_data(dataset_list, system_config, 
                                     dataset_colunms, verbose=verbose)


        # Verifica se o conjunto de dados com todos os dados da
        # aplicação não foi gerado.
        if dataset is None:
          # Como o conjunto de dados não foi gerado com sucesso, informa
          # o erro e passamos para o treinamento da próxima aplicação.
          print("➖➡️ Erro ao gerar o preditor para a aplicação "
                f"{application_name}! Não foi possível ler os conjuntos de "
                "dados!")

          # Como um erro ocorreu, mudamos status para False.
          status = False

          # Como não podemos mais continuar o processo de construção do
          # preditor para essa aplicação, passamos para a próxima 
          # aplicação.
          continue
      
        # Como foi possível ler com sucesso o conjuento de dados da
        # aplicação, vamos agora filtrar os outliers do conjunto de 
        # dados, usando a função filter_training_data, que retornará um
        # novo DataFrame com o conjunto de dados. O retorno da função
        # será armazenado em clean_dataset, e será uma referência para
        # o DataFrame do conjunto de dados filtrado se a filtragem for
        # feita sem erros, ou None caso algum erro ocorra durante a
        # flitragem.
        clean_dataset = filter_training_data(dataset, input_variables, 
                                             filter_variables, outlier_limit, 
                                             verbose=verbose)

        # Verifica se o conjunto de dados flitrado a partir do conjunto
        # de dados referenciado por dataset não foi gerado.
        if clean_dataset is None:
          # Como o conjunto de dados filtrado não foi gerado com
          # sucesso, informa o erro e passamos para o treinamento da
          # próxima aplicação.
          print("➖➡️ Erro ao gerar o preditor para a aplicação "
                f"{application_name}! Não foi possível filtrar o conjunto de "
                "dados concatenado!")

          # Como um erro ocorreu, mudamos status para False.
          status = False

          # Como não podemos mais continuar o processo de construção do
          # preditor para essa aplicação, passamos para a próxima 
          # aplicação.
          continue

        # Agora vamos usar a função optimize_hyperparameters para
        # otimizar os hiperparâmetros selecionados, ou seja, os 
        # definidos no arquivo de configuração de cada modelo, sendo os
        # modelos defindos pelo dicionário referenciado pela chave 
        # 'models' do dicionário com o arquivo de configuração do
        # teinamento, e sendo cada chave do dicionário referenciado pela
        # chave 'models' o dicinário com as informações necessárias para
        # treinar o modelo, como os hiperparâmetros a serem otimizados e
        # os possíveis valores para cada hiperparâmetro. A função 
        # retona os seguintes dicionários descitos a seguir e cujas 
        # refrências são armazenadas nas variáveis indicadas, ou None
        # para ambos os retornos se algum erro ocorrer durante a 
        # otimização dos hiperparâmetros de um modelo:
        #
        # initialized_models: Dicionário em que cada chave é o nome 
        #                     interno do modelo, ou seja a chave no
        #                     dicionário referenciado pela chave 
        #                     'models' do dicionário com as 
        #                     configurações gerais de treinamento, e o 
        #                     valor da chave uma referência a um objeto
        #                     do modelo já inicialiado com os melhores
        #                     valores dos hiperparâmetros e também com  
        #                     os hiperparâmetros com valores fixos, se
        #                     existirem. Este dicionário será usado pela
        #                     fase de escolha do melhor nodelo.
        # best_params_models: Dicionário em que novamente cada chave é
        #                     o nome interno do modelo, e o valor é um
        #                     dicionpario em que cada chave é um dos
        #                     hiperparâmetros, os otimizados e os fixos
        #                     se forem definidos, e o valor da chave de
        #                     um hiperparâmetro é o valor desse 
        #                     hiperparâmetro. Este dicionário será usado
        #                     pela função que efetivamente cria o 
        #                     preditor, ou seja, treina o modelo auxiliar
        #                     usado para escolher a melhor sugestão de
        #                     configuração e, se definido, o modelo para
        #                     predizer o tempo da aplicação quando
        #                     executa com esta sugestão.
        initialized_models, best_params_models = optimize_hyperparameters(
          clean_dataset,
          models,
          suggestion_variables,
          application_variables,
          group_variables,
          suggestion_prediction_variable,
          verbose
        )

        # Verifica se ocorreu algum erro ao otimizar os hiperparâametros
        # dos modelos. Como tanto initialized_models e 
        # best_params_models serão None, basta veriicar se uma delas é
        # None. Foi escolhida a initialized_models, mas poderia ser
        # também a best_params_models.
        if initialized_models is None:
          # Como não foi possível orimizar os hiperparâmertros dos 
          # modelos, informa o erro e passamos para o treinamento da
          # próxima aplicação.
          print("➖➡️ Erro ao gerar o preditor para a aplicação "
                f"{application_name}! Não foi possível otimizar os "
                "hiperparâmetros dos modelos!")

          # Como um erro ocorreu, mudamos status para False.
          status = False

          # Como não podemos mais continuar o processo de construção do
          # preditor para essa aplicação, passamos para a próxima 
          # aplicação.
          continue

        # Depois de otimizar com sucesso os hiperparâmetros dos modelos
        # a serem avaliados, vamos usar a função choose_best_model para
        # escolher o melhor desses modelos, que será usado ap criar o
        # preditor da aplicação. Nesta função, vamos usar o dicionário
        # initialized_models, que como vamos tem uma chave para cada
        # modelo a ser avaluado, que é nome interno desde modelo como
        # definido anteriormente, e cujo valor da chave é a referência
        # para o objeto do nodelo já inicializado com os melhores
        # valores para os hiperparâmetros e os valores dos 
        # hiperparâmetros fixos. A funçaõ retorna o seguinte, sendo que
        # ambos os retornos serão None se algum erro ocorrer durante a
        # avaliação e escolha do melhor modelo:
        #
        # best_model_name: Nome interno do melhor modelo, ou seja, o
        #                  modelo com a maior pontuação.
        # best_model_score: Pontuação do melhor modelo, usada somente 
        #                   em mensagens de vornosidade.
        best_model_name, best_model_score = choose_best_model(
          clean_dataset, 
          initialized_models, 
          suggestion_variables,
          application_variables, 
          group_variables, 
          suggestion_prediction_variable, 
          verbose
        ) 

        # Verifica se ocorreu algum erro ao escolher o meljor modelo. 
        # Como tanto best_model_name e best_model_score serão None, 
        # basta veriicar se uma delas é None. Foi escolhida a 
        # best_model_name, mas poderia ser também a best_model_score.
        if best_model_name is None:
          # Como não foi possível orimizar os hiperparâmertros dos 
          # modelos, informa o erro e passamos para o treinamento da
          # próxima aplicação.
          print("➖➡️ Erro ao gerar o preditor para a aplicação "
                f"{application_name}! Não foi possível escolher o melhor "
                "modelo!")

          # Como um erro ocorreu, mudamos status para False.
          status = False

          # Como não podemos mais continuar o processo de construção do
          # preditor para essa aplicação, passamos para a próxima 
          # aplicação.
          continue

        # Para faciliar o acesso, definimos a variável best_model_params
        # com o dicionário com os hiperparâmetros, tanto otimizados como
        # os fixos, para o modelo com o nome interno best_model_name. A
        # referência para ess dicionário estára associada à chave 
        # best_model_name do dicionário best_params_models.
        best_model_params = best_params_models[best_model_name]

        # Para faciliar o acesso, definimos a variável best_model_info
        # com a referência para o dicionário com as informações de
        # treinamento do modlo com o nome interno best_model_name. Como
        # vimos anteriormente, best_model_name é uma chave do dicionário
        # refecenciado pela chave 'models' do dicionário com as 
        # configurações gerais de treinamento.
        best_model_info = training_config["models"][best_model_name]

        # Criação do preditor.
        predictor = generate_application_predictor(clean_dataset, 
                                                   best_model_name, 
                                                   best_model_score, 
                                                   best_model_info,
                                                   best_model_params,
                                                   suggestion_variables,
                                                   application_variables,
                                                   group_variables,
                                                   prediction_variables,
                                                   verbose)

        best_model_acronym_name = models[best_model_name]['name']

        # Salvamento do preditor e atualiação do arquivo de configuração
        # dos preditores.
        status = save_application_predictor(best_model_name, 
                                            best_model_acronym_name,
                                            predictor, predictors_file_path, 
                                            predictors_info_config, 
                                            predictors_info_config_obj, 
                                            suggestion_prediction_variable, 
                                            application_name, verbose)

        if not status:
          print("❌ Erro ao salvar o novo preditor para a aplicação "
                f"{application_name}!")

      else:  
          print(f"⚠️ Ignorando aplicação desconhecida {application_name}!")

    return status
  
  except KeyError as e:
    if application_name is None:
      print("❌ Erro ao processar uma das estruturas indexadas por chave, a "
            f"chave {e.args[0]} não existe, quando inicializando os "
            "treinamentos das aplicações!")
    else:
      print("❌ Erro ao processar uma das estruturas indexadas por chave, a "
            f"chave {e.args[0]} não existe, quando gerando o preditor para a "
            f"aplicação {application_name}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
    return False
  except AttributeError as e:
    if application_name is None:
      print("❌ Erro ao acessar um dos objetos internos do script, um dos "
            "seus atributo não existe, quando inicializando os treinamentos "
            "das aplicações!")
    else:
      print("❌ Erro ao acessar um dos objetos internos do script, um dos "
            "seus atributo não existe, quando gerando o preditor para a "
            f"aplicação {application_name}!")
    print(f"❌ Erro gerado: '{e.args[0]}'")  
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
    return False
  except ValueError as e:
    if application_name is None:
      print("❌ Erro ao inicializar ou usar estruturas internas do script, "
            "quando inicializando os treinamentos das aplicações!")
    else:

      print("❌ Erro ao inicializar ou usar estruturas internas do script, "
            f"quando gerando o preditor para a aplicação {application_name}!")
    print(f"❌ Erro gerado: '{e.args[0]}'")  
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
    return False
  except Exception as e:
    if application_name is None:
      print("❌ Erro desconhecido ao inicializar os treinamentos das "
            "aplicações!")
    else:
      print("❌ Erro desconhecido ao gerar o preditor para a aplicação "
            f"{application_name}")
    print(f"❌ Parâmetros do erro: {e.args}!")
    print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
    return False

def execute_commands(command, applications_configs, training_config,
                     system_config, verbose):
 commands_dict = {
   'applications': partial(applications_command, applications_configs),
   'models': partial(models_command, training_config),
   'train': partial(train_command, command[1:], applications_configs, training_config, system_config, verbose=verbose)  
 } 

 if len(command) > 0 and command[0] in commands_dict.keys():
   Status = commands_dict[command[0]]()
   if not Status:
     print(f"Erro ao executar o comando {command[0]}!")
   return Status
 else:  
  if command:
    print(f"Comando {command[0]} inválido!")
  else:  
    print(f"O comando necessário não foi fornecido!")
  return False
    
# Lê as configuraçoes;
(
  configs_file_path, 
  system_config, 
  applications_configs, 
  training_config
)  = read_configs(DEBUG_CODE)

if any(
  v is None for v in [
    configs_file_path, 
    system_config,
    applications_configs,
    training_config
  ]
):
  print("❌ Erro ao ler uma das configurações!")
  exit(-1)

# Processa os parâmetros do script.
command, verbose, parser = process_script_args()

# Executa os comandos do script.
status = execute_commands(command, applications_configs, training_config, system_config, verbose)

# Verifica se ocorreu algum erro.
if not status:
  exit(-2)