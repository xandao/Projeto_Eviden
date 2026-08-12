import pandas as pd
from Utils.Suggestions import FilterOutliers, BestHiperparams, DiscoverBestModel, SuggestionsPredictor
from Utils.ReadConfigs import ReadSystemConfig, ReadApplicationsConfigs, ReadTrainingConfig, PredictorsInfoConfig
import importlib
import argparse
import sys
import textwrap
from pathlib import Path
from functools import partial
from Utils.Common import base_files_path_env_name, base_files_path, configs_files_dir, debug_code, CustomFormatter

def read_configs(verbose=False):
  # Lê os as variáveis gerais.
  if base_files_path is None:
    print(f"❌ Variável de ambiente {base_files_path_env_name} com o caminho da base dos scripts não foi definida")
    return None, None, None, None
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

  # Lê os parâmetros dos modelos escolhidos para avaliação;
  training_config_file_path = configs_file_path / 'training_config.json'
  training_config = ReadTrainingConfig(verbose).read_training_config(training_config_file_path)

  return configs_file_path, applications_configs, training_config, system_config

def process_script_args():
  # Inicializa o parser para verificar parâ,etros de aplicação
  parser = argparse.ArgumentParser(description="Script para treinar os modelos para todas as aplicações cujo uso será otimizado.", 
                                   usage="trainer [opções] Command [Parâmetros]", add_help=False, 
                                   formatter_class=CustomFormatter)

  # Adiciona as opções do script de treinamento.
  opcoes = parser.add_argument_group("Opções principais")
  ajuda = parser.add_argument_group("Ajuda")
  opcoes.add_argument("command", type=str, nargs='*', help=textwrap.dedent('''Comando a ser executado. Pode ser um dos seguintes comandos:

applications: lista todas as aplicações que podemos treinar os modelos.

models: lista os modelos que são avaliados quando os preditores forem gerados.

train app1, app2, ..., appn -> Faz todo o processo de treinamento, da filtragem dos dados, otimização dos
hiperparâmetros dos modelos, escolha do melhor modelo e treinamento deste melhor modelo com todos os
dados, sendo gerado um modelo para auxiliar a geração das sugestões e outro para predizer o tempo.

Cada aplicação da lista é considerada na ordem dada e os treinamentos são idependentes, ou seja,
passar a lista é equivalente a executar o script com o comando para cada aplicação isoladamente.

'''))
  opcoes.add_argument("-v", "--verbose", action="store_true", default=False, help="Habilita a verbosidade do script.")
  ajuda.add_argument("-h", "--help", action="help", help="Mostra esta mensagem de ajuda e sai.")

  # Processa os parâmetros da linha de comando
  args = parser.parse_args()

  verbose = args.verbose
  command = args.command

  return command, verbose, parser

def applications_command(applications_config):
  for application_name in sorted(applications_config.keys()):
    print(application_name)

def models_command(training_config):
  for model_name in training_config['models'].keys():
    print(model_name)

def train_command(applications_name, applications_config, training_config, system_config, verbose=False):
  # Define o caminho do diretório com os arquivos dos preditores e do arquivo de configuração.
  predictors_file_path = base_files_path / Path(system_config['predictors_path'])
  # Lê as configurações que associam cada preditor a aplicação correspondente,
  predictors_info_file_path = predictors_file_path / system_config["predictors_info_config_filename"]
  predictors_info_config_obj = PredictorsInfoConfig()
  predictors_info_config = predictors_info_config_obj.read_predictors_info_config(predictors_info_file_path)
  if predictors_info_config is None:
    return False
  

  for application_key in applications_name:
    if application_key in applications_config.keys():
      application_info = applications_config[application_key]
      if verbose:
        print(f"\n-> Treinando todos os modelos para a aplicação {application_info['name']}")	

      # Variáveis usadas np programa
      variaveis_de_entrada = list(set(application_info['suggestions_parameters']+
                                      application_info['application_parameters']+
                                      application_info['training']['group_parameters']))
      variaveis_do_filtro = application_info['training']['filter_parameters']
      variaveis_das_predicoes = application_info['estimated_parameters']
      variavel_predita_da_suggestao = variaveis_das_predicoes['suggestion']

      # Lê os dados.
      dados = pd.DataFrame()

      for nome_arquivo in application_info['training']['dataset_files']:
        nome_arquivo_completo = base_files_path / Path(system_config['dataset_path']) / nome_arquivo
        dados_arquivo = pd.read_csv(nome_arquivo_completo, usecols=variaveis_de_entrada+
                                                                   variaveis_do_filtro+
                                                                   list(variaveis_das_predicoes.values()))
        dados = pd.concat([dados, dados_arquivo])
      dados = dados.reset_index(drop=True)		

      if verbose:
        print("--> Conjunto de dados original da aplicação, antes da filtragem do outliers:")
        print("\n", dados.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")
        print("--> Dados estatísticos referentes ao conjunto de dados original:")
        print("\n", dados.describe().to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")

      # Filtra os dados.
      data_filter = FilterOutliers()
      dados_limpos = data_filter.Filter(dados, variaveis_de_entrada, variaveis_do_filtro, training_config['filter']['outlier_limit'])

      if verbose:
        print("--> Conjunto de dados da aplicação após a filtragem dos outliers:")
        print("\n", dados_limpos.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")
        print("---> Dados estatísticos referentes ao conjunto de dados filtrado:")
        print("\n", dados_limpos.describe().to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")
        print(f"--> Predições para a variável alvo {variavel_predita_da_suggestao}")

      predictor_hiperparams = {}

      for preditor_key, model_info in training_config["models"].items():
        if verbose:
          print(f"--> Otimizando o modelo {model_info['name']} usando os hiperparâmetros {model_info['grid_search_parms']} e a busca em grade:")
        # Descobre e importa o modelo de modo dinâmico.
        module_path, model_name = model_info['import_path'].rsplit(".", 1)
        # Importa e obtem dinamicamente o modelo.
        model_module = importlib.import_module(module_path)
        model = getattr(model_module, model_name)

        # Faz a otimização dos hiperparâmetros.
        best_hyper = BestHiperparams(verbose=debug_code)
        best_params, best_score = best_hyper.optimize(dados_limpos, application_info['suggestions_parameters'], 
                                                      application_info['application_parameters'], 
                                                      application_info['training']['group_parameters'], 
                                                      variavel_predita_da_suggestao, 
                                                      model() if model_info['fixed_params'] is None else model(**model_info['fixed_params']),
                                                      model_info['grid_search_parms'])

        if verbose:
          print(f"---> Modelo {model_info['name']}: Melhores hiperparâmetros -> {best_params}; Melhor score -> {best_score}")
          print("----> Dataframe com a avaliação de todas as combinações dos hiperparâmetros:")         
          hyperparams_score = best_hyper.get_hrperparams_scores() 
          print("\n", hyperparams_score.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")
      
        if model_info['fixed_params'] is not None:
            best_params = dict(**best_params, **model_info['fixed_params'])
        predictor_hiperparams[preditor_key] = model(**best_params) 

      # Determina o melhor modelo, usando a validacao cruzada.;
      if verbose:	
        print(f"--> Determinando o melhor modelo dentre os modelos da lista  {', '.join(predictor_hiperparams.keys())}, usando a validação cruzada com o LOGO:")
      cross_validator = DiscoverBestModel(verbose=debug_code)	
      best_model_name, best_model_score, results_df, mean_scores_models_df = cross_validator.best_model(dados_limpos, 
                                                                                  application_info['suggestions_parameters'], 
                                                                                  application_info['application_parameters'],  
                                                                                  application_info['training']['group_parameters'], 
                                                                                  variavel_predita_da_suggestao, predictor_hiperparams)
      if verbose:					
        print(f'---> Dataframe com os resultados das avaliações dos modelos:')
        print("\n", results_df.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")
        print(f'---> Dataframe com os resultados médios para cada modelo, ordenado do melhor para o pior modelo:')
        print("\n", mean_scores_models_df.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")
        print(f'--> Treinando agora o preditor com o melhor modelo {best_model_name} (pontuação: {best_model_score}), usando {best_params} como os hiperparâmetros customizados.')

      # Descobre e importa o modelo de modo dinâmico.
      module_path, model_name = training_config["models"][best_model_name]['import_path'].rsplit(".", 1)
      # Importa e obtem dinamicamente o modelo.
      model_module = importlib.import_module(module_path)
      model = getattr(model_module, model_name)

      predictor = SuggestionsPredictor()
      predictor.fit(dados_limpos, application_info['suggestions_parameters'], 
                                  application_info['application_parameters'], 
                                  application_info['training']['group_parameters'], 
                                  variaveis_das_predicoes, 
                                  model, best_params,
                                  verbose=debug_code)
      model_name = training_config['models'][best_model_name]['name']
      preditor_file_name = predictors_file_path / f"{application_info['name']}_{model_name}_{variavel_predita_da_suggestao}.pickle"
      if verbose:
        print('---> Dataframe do oráculo:')
        oracle_df = predictor.get_oracle()
        print("\n", oracle_df.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")
      # Obtém as imformações das importâncias, se o modelo as define  
      importances_df = predictor.get_importances(verbose=debug_code)
      if verbose and importances_df is not None:
        print('---> Dataframe com as importâncias do modelo:')
        print("\n", importances_df.to_markdown(tablefmt="grid", floatfmt=".2f"), "\n", sep="")
      elif verbose:
        print("---> O modelo não avalia as importâncias das características.")

      # Removendo o arquivo anterior do preditor da aplicação, se existir.
      if application_key in predictors_info_config:
        old_preditor_file_name = predictors_info_config[application_key]
        old_preditor_file_path = predictors_file_path / f"{old_preditor_file_name}"
        if verbose:
          print(f'--> Removendo o preditor antigo {old_preditor_file_name}')  
        old_preditor_file_path.unlink(missing_ok=True)
      
      # Salva o arquivo do preditor no formato .pickle.
      print(f'--> Salvando o preditor treinado como o modelo {best_model_name} (nome {model_name}) no arquivo {preditor_file_name}.')

      predictor.save_predictor(preditor_file_name)

      # Salva as informações do arquivo do modelo do preditor.
      if verbose:
        print(f'--> Salvndo a informação do caminhio do preditor {preditor_file_name} para a aplicação {application_key} no arquivo de configuração dos preditores.')
      predictors_info_config[application_key] = preditor_file_name.name
      predictors_info_config_obj.save_predictors_info_config(predictors_info_config)
    else:  
        print(f"⚠️  Ignorando aplicação desconhecida {application_key}!")

def execute_commands(command, applications_configs, training_config, system_config, verbose):
 commands_dict = {
   'applications': partial(applications_command, applications_configs),
   'models': partial(models_command, training_config),
   'train': partial(train_command, command[1:], applications_configs, training_config, system_config, verbose=verbose)  
 } 

 if len(command) > 0 and command[0] in commands_dict.keys():
   commands_dict[command[0]]()
   return True
 else:  
  if command:
    print(f"Comando {command[0]} inválido!")
  else:  
    print(f"O comando necessário não foi fornecido!")
  return False
    
# Processa os parâmetros do script.
command, verbose, parser = process_script_args()

# Lê as configuraçoes;
configs_file_path, applications_configs, training_config, system_config  = read_configs(verbose)
if applications_configs is None or training_config is None or system_config is None:
  print("❌ Erro ao ler uma das configurações!")
  exit(-1)
# Executa os comandos do script.

Status = execute_commands(command, applications_configs, training_config, system_config, verbose)

if not Status:
  parser.print_help()
  sys.exit(-1)