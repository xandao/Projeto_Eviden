import pandas as pd
import numpy as np
from Utils.Suggestions import SuggestionsPredictor
from Utils.ReadConfigs import ReadSystemConfig, ReadApplicationsConfigs, ReadUserConfig, PredictorsInfoConfig
import sys
import argparse
from pathlib import Path
import os
from functools import partial
import subprocess
from Utils.Common import debug_code, configs_file_path

def read_configs():
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
  if system_config is None:
    predictors_info_config = None
  else:
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
  types_map = {
    'integer': int,
    'floating-point': float,
    'string': str
  }

  return types_map.get(type_name, str)    

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

def convert_user_params(required_applicaion_params, conversions, application_configs_dir):
  # Dicionario com os dataframes para os mapeamentos (para evitar ler eles em cada mapeamento, se usados mais de uma vez)
  dataframe_map_dict = {}
  # Funções de conversão
  def copy_func(*args):
    #print("Alo 1")
    return getattr(required_applicaion_params, args[0])

  def filesize_func(*args):
    #print("Alo 2")
    return Path(getattr(required_applicaion_params, args[0])).stat().st_size

  def map_func(user_arg_name, *args):
    # O primeiro parâmetro é o nome do arquivo com o datafraame com os mapeamentos.
    dataframe_map_file_name = args[0]
    dataframe_map_full_path_name = Path(application_configs_dir) / dataframe_map_file_name
    if dataframe_map_file_name in dataframe_map_dict.keys():
      df_map = dataframe_map_dict[dataframe_map_file_name]
    else:
      df_map = pd.read_csv(dataframe_map_full_path_name) 
      #print(f"Dataframe de mapeamento {dataframe_map_file_name}: \n\n")
      #print(df_map.to_markdown(tablefmt="grid"))
      dataframe_map_dict[dataframe_map_file_name] = df_map

    # Cria a condicional para fazer a procura no dataframe de mapeamento.
    list_search = []
    for user_option in args[1:]:
      list_search.append(f"{user_option}.astype('str') == '{getattr(required_applicaion_params, user_option)}'")  

    str_search = ' and '.join(list_search)

    result_df = df_map.query(str_search)
    mapped_value = result_df.loc[0, user_arg_name]
    
    return mapped_value
  try:
    converted_user_params = {}
    for variable in conversions:
      conversion_info = conversions[variable]
      conversion_type = conversion_info[0]
      conversion_args = conversion_info[1:]
      conversion_types = {
        'copy': partial(copy_func, *conversion_args),
        'filesize': partial(filesize_func, *conversion_args),
        'map':  partial(map_func, variable, *conversion_args),
      }

      converted_user_params[variable] = conversion_types[conversion_type]()

    return converted_user_params
  except FileNotFoundError as e:
    print(f"File {e.filename} not Found: {e.strerror}")
    return None
  except PermissionError as e:
    print(f"Permission denied when accessing the file {e.filename}: {e.error}")
    return None
  except KeyError as e:
    print(f"Key error when processing option value : {', '.join(e.args)}")
    return None
  except Exception as e:
    print(f"Unknown error when processing option value: {', '.join(e.args)}!")
    return None


def generate_submission_script(template_file_path, template_params):
  # TODO: predicamos decidir como preencher os campos --partition, --time, --mem.
  # TODO: O --exclusive está fixo, pois não sei se é recomendado treinar um modelo com --exclusive 
  #       usar o --oversubscribe.

  def format_size(size_in_bytes):
    labels = ['B', 'K', 'M', 'G', 'T', 'P']
    label_index = 0

    # Continua divindo pór 1024 até encontrar a escala correta
    while size_in_bytes >= 1024 and label_index < len(labels) - 1:
      size_in_bytes = size_in_bytes / 1024.0
      label_index += 1

    # Retorna o tamanho formatado.
    return f"{np.ceil(size_in_bytes):.0f}{labels[label_index]}"
  
  def format_time(time_in_seconds):
    days = time_in_seconds // 86400
    hours = (time_in_seconds % 86400) // 3600
    minutes = ((time_in_seconds % 86400) % 3600) // 60
    seconds = time_in_seconds % 60
    prefix = f"{days}-" if days > 0 else ""
    return f"{prefix}{hours:02}:{minutes:02}:{seconds:02}"
          
  # Salva em uma string o conteúdo do arquivo de template.
  template_content = template_file_path.read_text(encoding="utf-8")

  # Altera o campo número de nós
  number_of_nodes = template_params['suggestion_params'].get('nodes', 1)
  template_content = template_content.replace("<<number_of_nodes>>", f"{number_of_nodes}")

  # Altera o campo =umero de processos
  number_of_process = template_params['suggestion_params'].get('process', 1)
  template_content = template_content.replace("<<number_of_process_per_node>>", f"{number_of_process}")

  # Altera o campo ntasks (igual ao produto de número de nós e número de processos por nó)
  template_content = template_content.replace("<<total_tasks>>", f"{number_of_nodes * number_of_process}")

  # Altera o campo número de threads
  number_of_threads = template_params['suggestion_params'].get('threads', 1)
  template_content = template_content.replace("<<threads_per_process>>", f"{number_of_threads}")

  # Altera o campo nme_do_job
  template_content = template_content.replace("<<job_name>>", template_params['job_name'])

  # Altera o campo dos outros par+ametros.
  template_content = template_content.replace("<<application_params>>", ' '.join(template_params['application_params']))
  
  # Altera a partição a ser usada
  template_content = template_content.replace("<<partition>>", f"{template_params['partition']}")

  # Altera o tempo máximo de execução
  template_content = template_content.replace("<<max_time>>", f"{format_time(template_params['max_time'])}")

  # Altera o tipo de execução, compartilhada (----oversubscribe) ou exclusiva (--exclusive).
  template_content = template_content.replace("<<execution_type>>", "exclusive" if template_params['exclusive'] else "oversubscribe")

  # Altera o uso máxumo de memória
  template_content = template_content.replace("<<max_memory>>", f"{format_size(template_params['max_memory'] * 1024)}")

  return template_content

def optimize_application(configs_file_path, system_config, applications_config, user_config, 
                         application_args, predictors_info_config, user_args):
  # Verifica se o usuário deseja somente listar as aplicações
  if user_args.list:
      for application_id in sorted(applications_config.keys()):
        if user_args.verbose:
          print(f"⚠️ Apllication identification: {application_id}, possible executable name(s): {', '.join(applications_config[application_id]['user']['executable_names'])}")
      return True
  else:
    # Caso não deseje listar as aplucações, precisamos fornecer uma aplicaçao, pois o usuário deseja otimizar o uso dos reursos.
    if not application_args:
      print("❌ An application, with its parameters, was not provided. For each application, some parameters are mandatory. See more details using the help, specifying the application name.")
      return False
    
    # Diretorio dos arquivos de configuração das aplicações.
    application_configs_dir_path = configs_file_path / 'applications'
        
    # Determina nome da aplicação
    application_name = application_args[0]
    application_id = None
    for application_id_aux in applications_config.keys():
      if application_name in applications_config[application_id_aux]['user']['executable_names']:
        application_id = application_id_aux
        break

    # Verifica se a apliucação existe
    if application_id is None:
      print(f"⚠️ Sorry, but optimization for application {application_name} is not currently supported.")  
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
            print(f"❌ Error processing option --{suggestion_name}!")
            return False
          custom_suggestions[applications_config[application_id]['user']['suggestions_map'][suggestion_name]] = custom_params
        else:
          print(f'⚠️ Igoring --{suggestion_name} not used by the application {application_name}!')  
    else:
      custom_suggestions = None

    # Processa os patâmetros usados pela aplicação para o preditor. 
    user_application_params = convert_user_params(required_applicaion_params, applications_config[application_id]['user']['conversions'], 
                                                  application_configs_dir_path)  
    if user_application_params is None:
      return False

    #print(user_params_dict)
    #print(custom_suggestions)
    
    # Lê o preditor usado para fazer a melhor sugestão dos parâmetros de execução da aplicação.
    predictor_path = Path(system_config['predictors_path']) / predictors_info_config[application_id]
    predictor = SuggestionsPredictor.load_predictor(predictor_path)
    suggestion = predictor.get_suggestion(user_application_params, custom_suggestions, verbose=user_args.verbose)

    # Cria o mapeamento reverso para a impressao
    reversed_suggestions_map = {v:k for k, v in applications_config[application_id]['user']['suggestions_map'].items()}
    suggestion_mapped = {reversed_suggestions_map[k]:v for k,v in suggestion['Suggestion'].items()}
    
    # Obtém o caminho do arquivo de template, se as opçoes. 
    if user_args.run or not user_args.suggestion:
      if user_args.verbose:
        SuggestionsPredictor.print_suggestion(suggestion, suggestion_map=reversed_suggestions_map, show_time=True, show_memory=True)

      #print(applications_config[application_id]['user']['slurm'])
      # Cria o dicionário com as informações para construir o script de submissão (fiz o dicionário para tornar a função
      # independente de como os parâmetros são gerados).
      list_partitions = applications_config[application_id]['user']['slurm']
      template_params = {
        'application_name': application_name,
        'suggestion_params': suggestion_mapped,
        'job_name':  application_name if user_args.jobname is None else user_args.jobname,
        'application_params': application_args[1:],
      }
      default_partition = np.argmax([partition['default'] for partition in list_partitions])
      if 'Time' in suggestion.keys():
        predicted_time = np.ceil(suggestion['Time'])
        valid_time_partitions = {pos for pos, partition in enumerate(list_partitions) if partition['max_time'] >= predicted_time}
        if valid_time_partitions:                                           
          # A partição escolhida será a com menor tempo máximo.
          pos_best_partition = np.nanargmin([partition['max_time'] if pos in valid_time_partitions else np.nan for pos, partition in enumerate(list_partitions)])
          partition_used = list_partitions[pos_best_partition]
        else:   
          partition_used = list_partitions[default_partition]

        # Verifica se a partução escolhida tem tempo suficiente para executar o trabalho.  
        if predicted_time > partition_used['max_time']:
          print(f"⚠️ Warning: Predicted time {predicted_time} is greather than {partition_used['max_time']} maximun partition {partition_used['partition']} execution time!")
      else:
        partition_used = list_partitions[default_partition]
        
        # Dá um pelo menos aviso se a o tempo, caso predito, for maior do que o tempo máximo da partição escolhida e/ou
        # se o uso de mamória, caso predito, for maior do que o uso de memória máximo da partição escolhida
#        if not predicted_memory is None and predicted_memory > partition_used['max_memory']:
#          print(f"⚠️ Warning: Predicted time {predicted_memory} is greather than {partition_used['max_memory']} maximun partition {partition_used['partition']} memory that can be allocated!")

      template_params['partition'] = partition_used['partition']
      template_params['max_time'] = partition_used['max_time']
      template_params['max_memory'] = partition_used['max_memory']  
      template_params['exclusive'] = partition_used['exclusive']  

      template_file_path = Path(system_config['templates_path']) / applications_config[application_id]['user']['script_template_name']
      template_content = generate_submission_script(template_file_path, template_params)

      if user_args.verbose:      
        print("Submission script: \n ")
        print(template_content)
        print()

      # Salva no arquivo passado como parâmetro ou o nome default definido no arquivo de cofiguração do usuário
      with open(user_args.script, "w", encoding="utf-8") as script_file:
        script_file.write(template_content)

      try:
        # Executa o sbatch se a opção -r ou --run foi usada
        submission_program = user_config["slurm"]["submission_program"]
        result = subprocess.run([submission_program, f"{user_args.script}"], capture_output=True, text=True, check=True)   
        if user_args.verbose:
          print(f"stdout of {submission_program} execution:\n\n")
          print(result.stdout)
          print(f"\n\nstderr of {submission_program} execution:\n\n")
          print(result.stderr)
      except subprocess.CalledProcessError as e:
        # This will print the actual error from the terminal command
        print("❌ Command failed!")
        print("   Exit code:", e.returncode)
        print("   Error message:", e.stderr)
      except FileNotFoundError:
        print(f"❌ Critical error: Program {submission_program} not found!")
    else:
      SuggestionsPredictor.print_suggestion(suggestion, suggestion_map=reversed_suggestions_map)

    return True

# Lê os arquivos de confoguraçã.o
configs_file_path, system_config, applications_configs, user_config, predictors_info_config = read_configs()

if system_config is None or applications_configs is None or user_config is None or predictors_info_config is None:
  exit(-1)

# Processa a linha de comando.
user_args, application_args, parser = process_script_args(user_config)
#print(script_args)
#print(application_args)

# Processa os parâmetros da linha de comando
status = optimize_application(configs_file_path, system_config, applications_configs, user_config, application_args, 
                              predictors_info_config, user_args)
if not status:
  parser.print_help()
  exit(-1)
