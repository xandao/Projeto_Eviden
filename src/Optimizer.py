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
from Utils.Common import base_files_path_env_name, base_files_path, configs_files_dir, debug_code
import textwrap
import tempfile
import re

def read_configs(verbose=False):
  # Lê as variáveis gerais.
  if base_files_path is None:
    print(f"❌ Variável de ambiente {base_files_path_env_name} com o caminho da base dos scripts não foi definida")
    return None, None, None, None, None
  else:
    configs_file_path = base_files_path / configs_files_dir
  system_config_file_path = configs_file_path / 'system_config.json'
  system_config = ReadSystemConfig(verbose).read_system_config(system_config_file_path)

  # Lê os parâmetros da aplicação
  if system_config is None:
    applications_configs = None
  else:    
    applications_configs_dir_path = configs_file_path / system_config['applications_path']
    applications_configs = ReadApplicationsConfigs(verbose).read_applications_config(applications_configs_dir_path)

  # Lê as configurações do script do usuário
  user_config_file_path = configs_file_path / 'user_config.json'
  user_config = ReadUserConfig(verbose).read_user_config(user_config_file_path)

  # Lê as configurações que associam cada preditor a aplicação correspondente,
  if system_config is None:
    predictors_info_config = None
  else:
    predictors_info_file_parh = base_files_path / Path(system_config["predictors_path"]) / system_config["predictors_info_config_filename"]
    predictors_info_config = PredictorsInfoConfig(verbose).read_predictors_info_config(predictors_info_file_parh)

  return configs_file_path, system_config, applications_configs, user_config, predictors_info_config

def process_script_args():
  parser = argparse.ArgumentParser(description="Script para escolher a melhor configuração para aplicações selecionadas.",
                                   usage="%(prog)s [opções] -- [executável da aplicação] [-h] [opções obrigatórias da aplicação] [outras opções da aplicação]",
                                   add_help=False, formatter_class=argparse.RawTextHelpFormatter)

  opcoes = parser.add_argument_group("Opções principais")
  ajuda = parser.add_argument_group("Ajuda")
  opcoes.add_argument("-r", "--run", action="store_true", default=False, help="Submete o script com a melhor configuração de execução.")
  opcoes.add_argument("-j", "--jobname", type=str, default=None, help="Nome do trabalho registrado no sistema de submissão")
  opcoes.add_argument("-s", "--script", type=str, default=None, help="Salva o script gerado em um arquivo.")
  opcoes.add_argument("-S", "--suggestion", action="store_true", default=False, 
                      help="Somente mostra a sugestão para os parâmetros do script.")
  opcoes.add_argument("-n", "--nodes", type=str, nargs="+", default=None, 
                      help=textwrap.dedent('''Lista com os possíveis números de nós, se a aplicação usa mulltiplos nós.
Usada conjuntamente com as opções -p e -t que terão os valores default se não usadas.
Cada elemento da lista está no formato i:e:s, onde i é o número inicial, f é o final e s é o passo.  
Pode-se omitir o i, que será igual a 1, o e, que será igual a i, e o s, que será igual a 1.
Default 1:1.
Exemplos: -n 1 2:10:2 -> Nós: 1, 2, 4, 6, 8, 10.
          -n :10:2    -> Nós: 1, 3, 5, 7, 9.
          -n 1:5      -> Nós: 1, 2, 3, 4, 5.                                                                 
                      '''))                      
  opcoes.add_argument("-p", "--process", type=str, nargs="+", default=None, 
                      help=textwrap.dedent('''Lista com os possíveis números de nós, se a aplicação usa mulltiplos nós.
Usada conjuntamente com as opções -n e -t que terão os valores default se não usadas.
Cada elemento da lista está no formato i:e:s, onde i é o número inicial, f é o final e s é o passo. 
Pode-se omitir o i, que será igual a 1, o e, que será igual a i, e o s, que será igual a 1.
Default 1:1.
Exemplos: -p 1 2:      -> Processos: 1, 2.
          -n 1 :3:1    -> Processos: 1, 2, 3.
          -n :3 6:12:3 -> Processos: 1, 2, 3, 6, 9, 12                                                                  
                      '''))
  opcoes.add_argument("-t", "--threads", type=str, nargs="+", default=None, 
                      help=textwrap.dedent('''Lista com os possíveis números de nós, se a aplicação usa mulltiplos nós.
Usada conjuntamente com as opções -n e -p que terão os valores default se não usadas.
Cada elemento da lista está no formato i:e:s, onde i é o número inicial, f é o final e s é o passo.  
Pode-se omitir o i, que será igual a 1, o e, que será igual a i, e o s, que será igual a 1.
Default 1:1.
Exemplos: -n 1 2:24:2 -> Threads: 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24
          -n 2 :24:8  -> Threads: 2, 24, 32, 40, 48.
          -n 2 24 48  -> Threads: 2, 24, 48.  
                      '''))
  opcoes.add_argument("-v", "--verbose", action="store_true", default=False, help="Habilita a verbosidade do script.")
  opcoes.add_argument("-l", "--list", action="store_true", default=False, help="Lista as aplicações cujas execuções podem ser otimizadas pelo script.")
  ajuda.add_argument("-h", "--help", action="help", help="Mostra esta mensagem de ajuda e sai")
  # Divide os parâmetros do script e da aplicação (separados por "--").
  application_param_separator = '--'

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
  options = sorted(list(set(options)))
  return options

def convert_user_params(required_applicaion_params, conversions, application_configs_dir):
  # Dicionario com os dataframes para os mapeamentos (para evitar ler eles em cada mapeamento, se usados mais de uma vez)
  dataframe_map_dict = {}
  # Funções de conversão
  def copy_func(*args):
    return getattr(required_applicaion_params, args[0])

  def filesize_func(*args):
    return Path(getattr(required_applicaion_params, args[0])).stat().st_size

  def map_func(user_arg_name, *args):
    # O primeiro parâmetro é o nome do arquivo com o datafraame com os mapeamentos.
    dataframe_map_file_name = args[0]
    dataframe_map_full_path_name = Path(application_configs_dir) / dataframe_map_file_name
    if dataframe_map_file_name in dataframe_map_dict.keys():
      df_map = dataframe_map_dict[dataframe_map_file_name]
    else:
      df_map = pd.read_csv(dataframe_map_full_path_name) 

      # TODO: Deixei esta depuração, habilitada pela variável de ambiente APPOPTIMIZER_DEBUG.
      # TODO: Podemos tirar todas as depurações no futuro.
      # TODO: Ínicio do código de depuração:
      if debug_code:
        print(f"➡️  Dataframe de mapeamento {dataframe_map_file_name}: \n\n")
        print(df_map.to_markdown(tablefmt="grid"))
      # TODO: Fim do código de depuração.
        
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
    print(f"❌ O arquivo {e.filename} nao foi encontrado.")
    print(f"❌ Por favor, avise o erro ao adistrador do sistema o erro: {e.strerror}!")
    return None
  except PermissionError as e:
    print(f"❌ Erro de permissão ao acessar o arquivo {e.filename}.")
    print(f"❌ Por favor, avise o erro ao adistrador do sistema o erro: {e.error}.")
    return None
  except IOError as e:
    print(f"❌ Erro de I/O ao ler o arquivo {e.filename}!")
    print(f"❌ Código do erro: {e.errno}; Mensagem: {e.strerror}!")
    print(f"❌ Por favor, reporte este erro ao adminstrador do sistema!")
    return None
  except KeyError as e:
    print(f"❌ Erro interno ao processar o valor da opção {', '.join(e.args)}.")
    print(f"❌ Por favor, reporte este erro ao adminstrador do sistema!")
    return None
  except Exception as e:
    print(f"❌ Erro desconhecido ao processar o valor da opção {', '.join(e.args)}")
    print(f"❌ Por favor, reporte este erro ao adminstrador do sistema!")
    return None

def generate_submission_script(template_file_path, template_params):
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

# JobName, JobID, [Parâmetros da Sugestão: RAxML, NAS -> "NNodes", "Processo p/ no", "Thread p/ proc.";
# RAxMl SSCAD -> [NNodes, Thread], [Parâmetros da Aplicação: RAxML -> Bootstrap. Arquivo, Tamanho;
# NAS -< Benchmark, Classe, Zone X, Zone Y, Iterações, Grid X, Grid Y, Grid Z], Tempo Predito, 
# Score: EDP Predito.
#
# Obs.: EDP Predito -> EDP predito da configuração sugerida -> menor EDP de todas as configurções avaliadas.
#
# Script de monitoração: Iria, de tempos em tempos, para cada aplicação, pegar cada coluna JobID e verificar
# se o job já terminou e, em caso, positivo, adicionar as informações relevantes obtidas pelo sacct aos
# dados do job.

def submission_log():
  pass  

def optimize_application(configs_file_path, system_config, applications_config, user_config, 
                         application_args, predictors_info_config, user_args):
  # Verifica se o usuário deseja somente listar as aplicações
  if user_args.list:
      for application_id in sorted(applications_config.keys()):
        print(f"➡️  Aplicação {application_id}, possíveis nomes para os executáveis: {', '.join(applications_config[application_id]['user']['executable_names'])}")
      return True
  else:
    # Caso não deseje listar as aplucações, precisamos fornecer uma aplicaçao, pois o usuário deseja otimizar o uso dos reursos.
    if not application_args:
      print("❌ Não foi fornecido o nome da aplicação a ser otimizada e os seus parâm,etros de execução.")
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
      print(f"⚠️  A otimização para a aplicação {application_name} ainda não é suportada!")  
      return False

    application_partitios_list = applications_config[application_id]['user']['slurm']
    names_application_partitioms = {partition['partition'] for partition in application_partitios_list}

    # Processa os parâmetros da aplicação.
    parser_application = argparse.ArgumentParser(description="Parser responsável pelos parâmetros da aplicação.", prog=application_name,
                                                 usage="Alo!")
    applicatiom_params = applications_config[application_id]['user']['user_options']
    for param in applicatiom_params.keys():
      parser_application.add_argument(*applicatiom_params[param]['params'], required=True, help=applicatiom_params[param]['help'], 
                                      type=get_type(applicatiom_params[param]['type']), dest=param)

    # Converte os argumentos da aplicação para o dicionário a ser usado pela função de predição.                                  
    required_applicaion_params, other_applicatios_params = parser_application.parse_known_args(application_args[1:])

    # Verifica se o usuário usou as opções número de nós, de processos por nó, e de threads por processo.
    # Primeiramemte verifica se o usuário definiu alguma das opções de configuração;
    use_custom_config = False
    for suggestion_name in applications_config[application_id]['user']['suggestions_map']:
      if not hasattr(user_args, suggestion_name):
        print(f"❌ A configuração necessária {suggestion_name} não existe nas opções do script para a aplcação {application_id}.")
        print(f"❌ Por favor, reporte este erro ao adminstrador do sistema!")
        return False
      elif getattr(user_args, suggestion_name) is not None:
        use_custom_config = True
    
    # Se o usuário definir pelo menos uma opção, usa a configuração customizada com as outras opções com valores defaault se não definidas
    # pelo usuário.    
    if use_custom_config:
      custom_suggestions = {}
      for suggestion_name in applications_config[application_id]['user']['suggestions_map']:
        suggestion_value = getattr(user_args, suggestion_name)
        custom_params = get_options_suggestion(suggestion_value)
        if custom_params is None:
          print(f"❌ Erro de sintaxe ao processar a opção --{suggestion_name} com o valor {suggestion_value}!")
          return False
        max_value_custom_params = max(custom_params)
        max_possible_value = max([partition[suggestion_name] for partition in application_partitios_list])
        if max_value_custom_params > max_possible_value:
          print(f"⚠️  Descantando todos os valores para a opção \033[31m{suggestion_name}\033[0m maiores do que {max_possible_value} "
                f"permitidos pelas possíveis partições \033[1;34m{', '.join(names_application_partitioms)}\033[0m da aplicação "
                f"{application_name}!")
          custom_params = [custom_value for custom_value in custom_params if custom_value <= max_possible_value]                 

        custom_suggestions[applications_config[application_id]['user']['suggestions_map'][suggestion_name]] = custom_params
    else:  
      custom_suggestions = None
    # Processa os patâmetros usados pela aplicação para o preditor. 
    user_application_params = convert_user_params(required_applicaion_params, applications_config[application_id]['user']['conversions'], 
                                                  application_configs_dir_path)  
    if user_application_params is None:
      return False

    # Lê o preditor usado para fazer a melhor sugestão dos parâmetros de execução da aplicação.
    predictor_path = base_files_path / Path(system_config['predictors_path']) / predictors_info_config[application_id]
    predictor = SuggestionsPredictor.load_predictor(predictor_path)
    suggestion = predictor.get_suggestion(user_application_params, custom_suggestions, verbose=debug_code)

    # Cria o mapeamento reverso para a impressao
    suggestion_map = applications_config[application_id]['user']['suggestions_map']
    reversed_suggestions_map = {v:k for k, v in suggestion_map.items()}
    suggestion_mapped = {reversed_suggestions_map[k]:v for k,v in suggestion['Suggestion'].items()}
    
    # Obtém o caminho do arquivo de template, se as opçoes. 
    if user_args.run or not user_args.suggestion:
      if user_args.verbose:
        SuggestionsPredictor.print_suggestion(suggestion, suggestion_map=reversed_suggestions_map, show_time=True,
                                              show_score=True, show_X=True, show_y_pred=True)

      # Cria o dicionário com as informações para construir o script de submissão (fiz o dicionário para tornar a função
      # independente de como os parâmetros são gerados).
      list_partitions = applications_config[application_id]['user']['slurm']
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
      for partition in list_partitions:
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
        partition_used = next([partition for partition in list_partitions if partition["dafault"]])
        # Verifica se o tempo da partição é maior do que o temṕo predito, e sá um aviso se isso ocorrer
        if predicted_time > partition_used['max_time']:
          print(f"⚠️  O tempo predito aproximado {predicted_time} é maior do que o tempo máximo {partition_used['max_time']} de execução da partição {partition_used['partition']}!")

      # define os dados para gerar o script de confuguração.
      template_params['partition'] = partition_used['partition']
      template_params['max_time'] = partition_used['max_time']
      template_params['max_memory'] = partition_used['max_memory']  
      template_params['exclusive'] = partition_used['exclusive']  


      template_file_path = base_files_path / Path(system_config['templates_path']) / applications_config[application_id]['user']['script_template_name']
      template_content = generate_submission_script(template_file_path, template_params)

      if user_args.verbose:      
        print("➡️  Script de submissão: \n")
        print(template_content)
        print()

      # Salva no arquivo passado como parâmetro ou o nome default definido no arquivo de cofiguração do usuário
      # Se o usuário não fornecer um nome pela opção -s ou --script, cria um arquivo temporário.
      if user_args.script is None and user_args.run:
        try:
          with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as temp:
              temp.write(template_content)
              script_file_name = temp.name 

          # TODO: Depois podemos remover, se necessário, este código de depuração.
          # TODO: Início.
          if debug_code:
            print(f"➡️  Arquivo temporário {script_file_name} criado para armazenar o script de submissão.")
          # TODO: Fim  
        except IOError as e:
          print(f"❌ Não foi possível criar o arquivo temporário. {e.filename}")
          print(f"❌ Código do erro: {e.errno}; Mensagem: {e.strerror}!")
          print(f"❌ Por favor, reporte este erro ao adminstrador do sistema!")
          return False
      else:
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
      try:
        if user_args.run:
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
      finally:
        # Se criamos um arquivo temporario, removemos depois de usarmos.
        if user_args.script is None and user_args.run:

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
              print(f"➡️  Não foi possível remover o arquivo {e.filename}")
              print(f"➡️  Código do erro: {e.errno}; Mensagem: {e.strerror}!")
    else:
      SuggestionsPredictor.print_suggestion(suggestion, suggestion_map=reversed_suggestions_map, show_score=user_args.verbose, 
                                            show_X=user_args.verbose, show_y_pred=user_args.verbose, show_time=user_args.verbose)

    return True

# Processa a linha de comando.
user_args, application_args, parser = process_script_args()

# Lê os arquivos de confoguraçã.o
configs_file_path, system_config, applications_configs, user_config, predictors_info_config = read_configs(verbose=user_args.verbose)

if system_config is None or applications_configs is None or user_config is None or predictors_info_config is None:
  print('❌ Erro ao processar um dos arquivos de configuração. Por favor, avise o erro ao suporte.')
  exit(-1)

# Processa os parâmetros da linha de comando
status = optimize_application(configs_file_path, system_config, applications_configs, user_config, application_args, 
                              predictors_info_config, user_args)
if not status:
#  parser.print_help()
  exit(-1)
