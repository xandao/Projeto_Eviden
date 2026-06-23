import pandas as pd
import numpy as np
from Utils.Suggestions import SuggestionsPredictor
from Utils.ReadConfigs import ReadSystemConfig, ReadApplicationsConfig, ReadUserConfig
from pprint import pprint
import sys
import argparse
from pathlib import Path
import os

def read_configs():
  # Lê a variável de ambiente com o caminho do diretório com as configurações
  configs_file_path = Path(os.getenv('APPOPTIMIZER_CONFIGS_DIR', '../configs'))

  # Lê as variáveis gerais.
  system_config_file_path = configs_file_path / 'system_config.json'
  system_config = ReadSystemConfig().read_system_config(system_config_file_path)
  #pprint(system_config)

  # Lê os parâmetros da aplicação
  application_configs_dir_path = configs_file_path / 'applications'
  applications_configs = ReadApplicationsConfig().read_applications_config(application_configs_dir_path)
  #pprint(applications_configs)

  # Lê as configurações do script do usuário
  user_config_file_path = configs_file_path / 'user_config.json'
  user_config = ReadUserConfig().read_user_config(user_config_file_path)
  #pprint(user_config)

  return configs_file_path, system_config, applications_configs, user_config

def process_script_args(user_config):
  parser = argparse.ArgumentParser(description="Test script to predict suggestion variables.")

  parser.add_argument("-r", "--run", action="store_true", default=False, help="Submit script with the best configurations.")
  parser.add_argument("-s", "--script", type=str, default=user_config.get('default_script_name', 'script.sh'), 
                      help="Save generated script in a file.")
  parser.add_argument("-S", "--suggestion", action="store_true", default=False, 
                      help="Only returns suggested execution params.")
  parser.add_argument("-n", "--nodes", type=str, nargs="*", default=None, help='List of possible nodes')
  parser.add_argument("-p", "--process", type=str, nargs="*", default=None, help='List of possible process')
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

  return user_args, application_args 

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


def optimize_application(system_config, applications_config, user_config, 
                         application_args, user_args):
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

    # Obtém o caminho do arquivo de template, se as opçoes. 
    if user_args.run or not user_args.suggestion:
      template_file_path = Path(system_config['templates_path']) / applications_config[application_id_aux]['user']['script_template_name']
      print(template_file_path)  

      # Salva em uma string o conteúdo do arquivo de template.
      template_content = template_file_path.read_text(encoding="utf-8")
      print(template_content)
    else:
      template_content = None

    # Processa os parâmetros da aplicação.
    parser_application = argparse.ArgumentParser(description="Parser application parameters", prog=application_name)
    applicatiom_params = applications_config[application_id_aux]['user']['user_options']
    for param in applicatiom_params.keys():
      parser_application.add_argument(*applicatiom_params[param]['params'], required=True, help=applicatiom_params[param]['help'], 
                                      type=get_type(applicatiom_params[param]['type']))
    #parser_application.print_help()
    #sys.exit(0)
    required_applicaion_params, other_applicatios_params = parser_application.parse_known_args(application_args[1:])
    #print(required_applicaion_params, other_applicatios_params)
    if not user_args.nodes is None or not user_args.process is None or not user_args.threads is None:
      custom_suggestions = {}
      for suggestion_name in user_config['suggestions_names']:
        #print(applications_config[application_id_aux]['user']['suggestions_map'][suggestion_name])
        if suggestion_name in applications_config[application_id_aux]['user']['suggestions_map'].keys():
          custom_suggestions[applications_config[application_id_aux]['user']['suggestions_map'][suggestion_name]] = get_options_suggestion(getattr(user_args, suggestion_name))
        else:
          print(f'Igored --{suggestion_name} no used by application {application_name}!')  
    else:
      custom_suggestions = None
    #print(user_args)
    #print(hasattr(user_args, 'nodes'))
    #print(hasattr(user_args, 'alex'))
    #print(get_options_suggestion(user_args.nodes))
    #print(get_options_suggestion(user_args.process))
    #print(get_options_suggestion(user_args.threads))
    #print(custom_suggestions)
    
    # Converte os parâmetros da aplicação.

    # Lê o preditor para fazer a sugestão
      
    return True

# Lê os arquivos de confoguraçã.o
configs_file_path, system_config, applications_configs, user_config = read_configs()

# Processa a linha de comando.
user_args, application_args = process_script_args(user_config)
#print(script_args)
#print(application_args)

# Processa os parâmetros da linha de comando
status = optimize_application(system_config, applications_configs, user_config, application_args, user_args)

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