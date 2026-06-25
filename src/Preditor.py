import pandas as pd
import numpy as np
from Utils.Suggestions import SuggestionsPredictor
from Utils.ReadConfigs import ReadSystemConfig, ReadApplicationsConfigs, ReadUserConfig, PredictorsInfoConfig
from pprint import pprint
import sys
import argparse
from pathlib import Path
import os
from functools import partial

def read_configs():
  # Lê a variável de ambiente com o caminho do diretório com as configurações
  configs_file_path = Path(os.getenv('APPOPTIMIZER_CONFIGS_DIR', '../configs'))

  # Lê as variáveis gerais.
  system_config_file_path = configs_file_path / 'system_config.json'
  system_config = ReadSystemConfig().read_system_config(system_config_file_path)

  # Lê os parâmetros da aplicação
  application_configs_dir_path = configs_file_path / 'applications'
  applications_configs = ReadApplicationsConfigs().read_applications_config(application_configs_dir_path)

  # Lê as configurações do script do usuário
  user_config_file_path = configs_file_path / 'user_config.json'
  user_config = ReadUserConfig().read_user_config(user_config_file_path)

  # Lê as configurações que associam cada preditor a aplicação correspondente,
  predictors_info_file_parh = Path(system_config["predictors_path"]) / system_config["predictors_info_config_filename"]
  predictors_info_config = PredictorsInfoConfig().read_predictors_info_config(predictors_info_file_parh)

  return configs_file_path, system_config, applications_configs, user_config, predictors_info_config

def process_script_args(user_config):
  parser = argparse.ArgumentParser(description="Test script to predict suggestion variables.")

  parser.add_argument("-r", "--run", action="store_true", default=False, help="Submit script with the best configurations.")
  parser.add_argument("-j", "--jobname", type=str, default=None, help="Job name (used with option -r or --run)")
  parser.add_argument("-s", "--script", type=str, default=user_config.get('default_script_name', 'script.sh'), 
                      help="Save generated script in a file.")
  parser.add_argument("-S", "--suggestion", action="store_true", default=False, 
                      help="Only returns suggested execution params.")
  if "nodes" in user_config["suggestions_names"]:
    parser.add_argument("-n", "--nodes", type=str, nargs="*", default=None, help='List of possible nodes')
  if "process" in user_config["suggestions_names"]:
    parser.add_argument("-p", "--process", type=str, nargs="*", default=None, help='List of possible process')
  if "threads" in user_config["suggestions_names"]:
    parser.add_argument("-t", "--threads", type=str, nargs="*", default=None, help='List of possible threads')
  parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Enable output verbosity")
  parser.add_argument("-l", "--list", action="store_true", default=False, help="List available applications to optimize.")

  # Divide os parâmetros do script e da aplicação (separados por "--").
  application_param_separator = user_config.get('application_params_separator', '--')

  if application_param_separator in sys.argv:
    separator_pos = sys.argv.index(application_param_separator)
    script_args = sys.argv[1:separator_pos]
    application_args = sys.argv[separator_pos+1:]
  else:
    script_args = sys.argv[1:]
    application_args = []

  user_args = parser.parse_args(script_args)

  return user_args, application_args, parser

def get_type(type_name):
  match type_name:
    case 'integer':
      return int
    case 'floating-point':
      return float
    case 'string':
      return str
    case _:
      return str
    
def get_options_suggestion(suggestion_args):
  options = []
  if suggestion_args is None:
    suggestion_args = ['1:1']
  for arg in suggestion_args:
    try:
      # Parametro start:end:step 
      # Se não tem startm start=1
      # Se não tem end, end=start+1
      # Se não tem step, step=1
      if ':' in arg:
        params = arg.split(':')
        # Inicio da faixa.
        strstart = params[0].strip()
        strend = params[1].strip()
        if len(params) > 2:
          strstep = params[2].strip()
        else:
          strstep = "1"
        # Converte strstart para inteiro.
        if len(strstart) > 0:
          start = int(strstart) 
        else:
          start = 1   
        # Converte strend para inteiro
        if len(strend) > 0:
          end = int(strend)+1 
        else:
          end = start+1   
        # Converte strstep para inteiro
        if len(strstep) > 0:
          step = int(strstep) 
        else:
          step = 1
        options.extend(range(start, end, step))  
      else:
        param = int(arg)
        options.append(param)
    except Exception as e:
      return None
  
  # Remove valores duplicados
  options = list(set(options))
  return options

def convert_user_params(required_applicaion_params, conversions):
  # Funções de conversão
  def copy_func(*args):
    return getattr(required_applicaion_params, args[0])

  def filesize_func(*args):
    return Path(getattr(required_applicaion_params, args[0])).stat().st_size

  def map_func(map_info, *args):
    aux_info = args[-1]
    args = args[:-1]
    while len(args) > 0:
      aux_info = aux_info[getattr(required_applicaion_params(args[0]))]
      args = args[1:]
    return aux_info  

  try:
    converted_user_params = {}
    for variable in conversions:
      conversion_info = conversions[variable]
      conversion_type = conversion_info[0]
      conversion_args = conversion_info[1:]
      conversion_types = {
        'copy': partial(copy_func, *conversion_args),
        'filesize': partial(filesize_func, *conversion_args),
        'map':  partial(map_func, *conversion_args),
      }

      converted_user_params[variable] = conversion_types[conversion_type]()

    return converted_user_params
  except FileNotFoundError:
    print(f"File {getattr(required_applicaion_params, conversion_args[0])} not Found!")
    return None
  except PermissionError:
    print(f"Permission denied when accessing the file {getattr(required_applicaion_params, conversion_args[0])}")
    return None
  except Exception:
    print(f"Error when processing option value!")
    return None

def generate_submission_script(user_config, user_args, template_file_path, application_name, 
                               application_params, suggestion_params):
  print(template_file_path)  

  # Salva em uma string o conteúdo do arquivo de template.
  template_content = template_file_path.read_text(encoding="utf-8")

  # Altera o campo número de nós
  number_of_nodes = suggestion_params.get('nodes', 1)
  template_content = template_content.replace("<<number_of_nodes>>", f"{number_of_nodes}")

  # Altera o campo =umero de processos
  number_of_process = suggestion_params.get('process', 1)
  template_content = template_content.replace("<<number_of_process_per_node>>", f"{number_of_process}")

  # Altera o campo ntasks (igual ao produto de número de nós e número de processos por nó)
  template_content = template_content.replace("<<total_tasks>>", f"{number_of_nodes * number_of_process}")

  # Altera o campo número de threads
  number_of_threads = suggestion_params.get('threads', 1)
  template_content = template_content.replace("<<threads_per_process>>", f"{number_of_threads}")

  # Altera o campo nme_do_job
  if user_args.jobname is None:
    template_content = template_content.replace("<<job_name>>", application_name)
  else:  
    template_content = template_content.replace("<<job_name>>", user_args.jobname)

  # Altera o campo dos outros par+ametros.
  template_content = template_content.replace("<<application_params>>", ' '.join(application_params))
  
  # TODO: Altera os campos que ainda não sei como ontê-los (coloquei temporariamente na 
  # configuração do script.

  # Altera a partição a ser usada
  template_content = template_content.replace("<<partition>>", f"{user_config['slurm']['partition']}")

  # Altera o tempo máximo de execução
  template_content = template_content.replace("<<max_time>>", f"{user_config['slurm']['max_time']}")

  # Altera o uso máxumo de memória
  template_content = template_content.replace("<<max_memory>>", f"{user_config['slurm']['max_memory']}")

  print(template_content)

  return template_content


def optimize_application(system_config, applications_config, user_config, 
                         application_args, predictors_info_config, user_args):
  # Verifica se o usuário deseja somente listar as aplicações
  if user_args.list:
      for application_id in sorted(applications_config.keys()):
        print(f"Apllication identification: {application_id}, possible executable name(s): {', '.join(applications_config[application_id]['user']['executable_names'])}")
      return True
  else:
    # Caso não deseje listar as aplucações, precisamos fornecer uma aplicaçao, pois o usuário deseja otimizar o uso dos reursos.
    if not application_args:
      print("An application, with its parameters, was not provided. For each application, some parameters are mandatory. See more details using the help, specifying the application name.")
      return False
    
    # Processa os parâmetros possíveis nós, processos por nó e threads.
    custom_configuratios = None  
    
    # Determina nome da aplicação
    application_name = application_args[0]
    application_id = None
    for application_id_aux in applications_config.keys():
      if application_name in applications_config[application_id_aux]['user']['executable_names']:
        application_id = application_id_aux
        break

    # Verifica se a apliucação existe
    if application_id is None:
      print(f"Sorry, but optimization for application {application_name} is not currently supported.")  
      return False

    # Processa os parâmetros da aplicação.
    parser_application = argparse.ArgumentParser(description="Parser application parameters", prog=application_name)
    applicatiom_params = applications_config[application_id]['user']['user_options']
    for param in applicatiom_params.keys():
      parser_application.add_argument(*applicatiom_params[param]['params'], required=True, help=applicatiom_params[param]['help'], 
                                      type=get_type(applicatiom_params[param]['type']), dest=param)

    # Converte os argumentos da aplicação para o dicionário a ser usado pela função de predição.                                  
    #parser_application.print_help()
    #sys.exit(0)
    required_applicaion_params, other_applicatios_params = parser_application.parse_known_args(application_args[1:])
    #print(required_applicaion_params, other_applicatios_params)
    if not user_args.nodes is None or not user_args.process is None or not user_args.threads is None:
      custom_suggestions = {}
      for suggestion_name in user_config['suggestions_names']:
        #print(applications_config[application_id_aux]['user']['suggestions_map'][suggestion_name])
        if suggestion_name in applications_config[application_id]['user']['suggestions_map'].keys():
          custom_params = get_options_suggestion(getattr(user_args, suggestion_name))
          if custom_params is None:
            print(f"Error processing option --{suggestion_name}!")
            return False
          custom_suggestions[applications_config[application_id]['user']['suggestions_map'][suggestion_name]] = custom_params
        else:
          print(f'Igored --{suggestion_name} no used by application {application_name}!')  
    else:
      custom_suggestions = None

    # Processa os patâmetros usados pela aplicação para o preditor. 
    user_application_params = convert_user_params(required_applicaion_params, applications_config[application_id]['user']['conversions'])  
    if user_application_params is None:
      return False

    #print(user_params_dict)
    #print(custom_suggestions)
    
    # Lê o preditor usado para fazer a melhor sugestão dos parâmetros de execução da aplicação.
    predictor_path = Path(system_config['predictors_path']) / predictors_info_config[application_name]
    predictor = SuggestionsPredictor.load_predictor(predictor_path)
    suggestion = predictor.get_suggestion(user_application_params, custom_suggestions, verbose=user_args.verbose)

    # Cria o mapeamento reverso para a impressao
    reversed_suggestions_map = {v:k for k, v in applications_config[application_id]['user']['suggestions_map'].items()}
    suggestion_mapped = {reversed_suggestions_map[k]:v for k,v in suggestion['Suggestion'].items()}
    
    # Obtém o caminho do arquivo de template, se as opçoes. 
    if user_args.run or not user_args.suggestion:
      if user_args.verbose:
        SuggestionsPredictor.print_suggestion(suggestion, suggestion_map=reversed_suggestions_map)
      template_file_path = Path(system_config['templates_path']) / applications_config[application_id]['user']['script_template_name']
      template_content = generate_submission_script(user_config, user_args, template_file_path, application_name, application_args[1:], 
                                                    suggestion_mapped)
    else:
      SuggestionsPredictor.print_suggestion(suggestion, suggestion_map=reversed_suggestions_map)


    return True

# Lê os arquivos de confoguraçã.o
configs_file_path, system_config, applications_configs, user_config, predictors_info_config = read_configs()

# Processa a linha de comando.
user_args, application_args, parser = process_script_args(user_config)
#print(script_args)
#print(application_args)

# Processa os parâmetros da linha de comando
status = optimize_application(system_config, applications_configs, user_config, application_args, predictors_info_config, user_args)
if not status:
  parser.print_help()
  exit(-1)
#parser.add_argument("-N", type=int, required=True, default=None, help="Bootstrap value")
#
#parser.add_argument("-s", type=str, required=True, help="Input file path")
#
#parser.add_argument("-v", "--verbose", action="store_true", default=False, help="Enable output verbosity")
#
## Parse the parameters
#args = parser.parse_args()
#
#variaveis_de_entrada = ['NNodes', 'Processo p/ no', 'Thread p/ proc.', 'Bootstrap', 'Tamanho']
#variaveis_de_saida = ['ElapsedRaw', 'Consumo de Energia Total (J)', 'EDP']
#
#bootstrap = args.N
#caminho_arquivo = Path(args.s)
#tamanho = caminho_arquivo.stat().st_size
#verbose = args.verbose
#
#user_app_teste = {'Bootstrap': bootstrap, 'Tamanho': tamanho}
#
#print("Lendo o preditor")
#
#nome_arquivo_preditor = '../predictors/EDP_ExtraTreesRegressor_raxml.pickle'
#
#predictor = SuggestionsPredictor.load_predictor(nome_arquivo_preditor)
#
#print("Teste 1: Teste usando as configurações padrões (as mesmas do treinamento)")
#
#suggestion = predictor.get_suggestion(user_app_teste, verbose=verbose)
#
#print(f"Sugestão para o bootstap {bootstrap} e o tamanho do arquivo {tamanho} (arquivo {args.s})")
#SuggestionsPredictor.print_suggestion(suggestion, show_score=True, show_X=True, show_y_pred=True)
#
#custom_configuratios = {
#	'NNodes' : range(1, 11), 
#	'Processo p/ no': [1, 2, 4], 
#	'Thread p/ proc.': [2, 4, 8, 16, 32, 64],
#}
#custom_suggestion = predictor.get_suggestion(user_app_teste, custom_configuratios, verbose=verbose)
#
#print(f"Teste 2: Teste usando as configurações passadas pelo usuário:")
#pprint(custom_configuratios)
#print(f"Sugestão para o bootstap {bootstrap} e o tamanho do arquivo {tamanho} (arquivo {args.s})")
#SuggestionsPredictor.print_suggestion(custom_suggestion, show_score=True, show_X=True, show_y_pred=True)
#
#user_app_teste_list = pd.DataFrame({'Bootstrap': [50, 500, 100, 1000], 'Tamanho': [123456, 654321, 231143, 198574]})
#
#print("Testes 3 e 4, iguais ao 1 e 2, mas usando os parâmetros do usuário dados no seguinte dataframe:")
#print(user_app_teste_list.to_markdown(tablefmt="grid"))
#print("Teste 3: Teste usando as configurações padrões (as mesmas do treinamento):")
#
#suggestions = predictor.get_suggestions(user_app_teste_list, verbose=verbose)
#
#for pos, suggestion in enumerate(suggestions):
#	print(f"Sugestão para o bootstap {user_app_teste_list.loc[pos, 'Bootstrap']} e o tamanho do arquivo {user_app_teste_list.loc[pos, 'Tamanho']})")
#	SuggestionsPredictor.print_suggestion(suggestion, show_score=True, show_X=True, show_y_pred=True)
#  
#custom_suggestions = predictor.get_suggestions(user_app_teste_list, custom_configuratios, verbose=verbose)
#
#print(f"Teste 4: Teste usando as configurações passadas pelo usuário:")
#for pos, suggestion in enumerate(custom_suggestions):
#	print(f"Sugestão para o bootstap {user_app_teste_list.loc[pos, 'Bootstrap']} e o tamanho do arquivo {user_app_teste_list.loc[pos, 'Tamanho']})")
#	SuggestionsPredictor.print_suggestion(suggestion, show_score=True, show_X=True, show_y_pred=True)