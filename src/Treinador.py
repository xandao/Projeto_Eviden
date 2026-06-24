import pandas as pd
from Utils.Suggestions import FilterOutliers, BestHiperparams, DiscoverBestModel, SuggestionsPredictor
from Utils.ReadConfigs import ReadSystemConfig, ReadApplicationsConfigs, ReadTrainingConfig
import importlib
import argparse
import os
import sys
from pathlib import Path
from functools import partial

def read_configs():
  # Lê a variável de ambiente com o caminho do diretório com as configurações
  configs_file_path = Path(os.getenv('APPOPTIMIZER_CONFIGS_DIR', '../configs'))
  # Lê os as variáveis gerais.
  system_config_file_path = configs_file_path / 'system_config.json'
  system_config = ReadSystemConfig().read_system_config(system_config_file_path)
  #pprint(system_config)

  # Lê os parâmetros da aplicação
  application_configs_dir_path = configs_file_path / 'applications'
  applications_configs = ReadApplicationsConfigs().read_applications_config(application_configs_dir_path)
  #pprint(applications_configs)

  # Lê os parâmetros dos modelos escolhidos para avaliação;
  training_config_file_path = configs_file_path / 'training_config.json'
  training_config = ReadTrainingConfig().read_training_config(training_config_file_path)
  #pprint(training_config)
  return configs_file_path, applications_configs, training_config, system_config

def process_script_args():
  # Inicializa o parser para verificar parâ,etros de aplicação
  parser = argparse.ArgumentParser(description="Script to train and generate models for the applications described configuration files")

  # Add a boolean switch flag (true/false switch)
  parser.add_argument("-v", "--verbose", action="store_true",default=False, help="Increase output verbosity")
  parser.add_argument("command", type=str, nargs='*', help="Command: can be applications, to list allaplications, models to list the models, or train to generete the best model for a selected application")

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

def train_command(applications_name, applications_config, training_config, configs_file_path, system_config, verbose=False):
  for application_key in applications_name:
    if application_key in applications_config.keys():
      application_info = applications_config[application_key]
      if verbose:
        print(f"-> Training all models for the application {application_info["name"]}")	

      # Variáveis usadas np programa
      variaveis_de_entrada = list(set(application_info['suggestions_parameters']+
                                      application_info['application_parameters']+
                                      application_info['training']['group_parameters']))
      variaveis_do_filtro = application_info['training']['filter_parameters']
      variavel_de_saida = application_info['estimated_parameter']

      # Lê os dados.
      dados = pd.DataFrame()

      for nome_arquivo in application_info['training']['dataset_files']:
        nome_arquivo_completo = Path(system_config['dataset_path']) / nome_arquivo
        dados_arquivo = pd.read_csv(nome_arquivo_completo, usecols=variaveis_de_entrada+
                                                                   variaveis_do_filtro+
                                                                   [variavel_de_saida])
        dados = pd.concat([dados, dados_arquivo])
      dados = dados.reset_index(drop=True)		

      if verbose:
        print("--> Original application data:\n")
        print(dados.to_markdown(tablefmt="grid"))
        print("--> Static information of original application data:\n")
        print(dados.describe().to_markdown(tablefmt="grid", floatfmt=".2f"))

      # Filtra os dados.
      data_filter = FilterOutliers()
      dados_limpos = data_filter.Filter(dados, variaveis_de_entrada, variaveis_do_filtro, training_config['filter']['outlier_limit'])

      if verbose:
        print("--> Filtered application data:\n")
        print(dados_limpos.to_markdown(tablefmt="grid"))
        print("--> Static information of filtered application data:\n")
        print(dados_limpos.describe().to_markdown(tablefmt="grid", floatfmt=".2f"))
        print(f'--> Predictions for the target {variavel_de_saida}')

      predictor_hiperparams = {}

      for preditor_key, model_info in training_config["models"].items():
        if verbose:
          print(f'--> Optimizing model {model_info["name"]} hyperparameters:')
        # Descobre e importa o modelo de modo dinâmico.
        module_path, model_name = model_info['import_path'].rsplit(".", 1)
        # Importa e obtem dinamicamente o modelo.
        model_module = importlib.import_module(module_path)
        model = getattr(model_module, model_name)

        # Faz a otimização dos hiperparâmetros.
        best_hyper = BestHiperparams()
        best_params, best_score = best_hyper.optimize(dados_limpos, application_info['suggestions_parameters'], 
                                                      application_info['application_parameters'], 
                                                        application_info['training']['group_parameters'], 
                                                        variavel_de_saida, 
                                                        model() if model_info['fixed_params'] is None else model(**model_info['fixed_params']),
                                                        model_info['grid_search_parms'],
                                                        verbose=verbose)

        if verbose:
          print(f"---> Model {model_info['name']}: Best hyperparameters -> {best_params}; Best score -> {best_score}")
      
        if not model_info['fixed_params'] is None:
            best_params = dict(**best_params, **model_info['fixed_params'])
        predictor_hiperparams[preditor_key] = model(**best_params) 

      # Determina o melhor modelo, usando a validacao cruzada.;
      if verbose:	
        print('--> Discover the best model of {}')
      cross_validator = DiscoverBestModel()	
      best_model_name, best_model_score, results_df, mean_scores_models_df = cross_validator.best_model(dados_limpos, 
                                                                                  application_info['suggestions_parameters'], 
                                                                                  application_info['application_parameters'],  
                                                                                  application_info['training']['group_parameters'], 
                                                                                  variavel_de_saida, predictor_hiperparams,
                                                                                  verbose=verbose)
      if verbose:					
        print(f'--> Dataframe with the results of models evaluation')
        print(results_df.to_markdown(tablefmt="grid"))
        print(f'--> Dataframe with the mean results of models evaluation')
        print(mean_scores_models_df.to_markdown(tablefmt="grid"))
        print(f'--> Trainining predictor with best model {best_model_name} (score: {best_model_score}), using {best_params} as hiperparameters.')

      # Descobre e importa o modelo de modo dinâmico.
      module_path, model_name = training_config["models"][best_model_name]['import_path'].rsplit(".", 1)
      # Importa e obtem dinamicamente o modelo.
      model_module = importlib.import_module(module_path)
      model = getattr(model_module, model_name)

      predictor = SuggestionsPredictor()
      predictor.fit(dados_limpos, application_info['suggestions_parameters'], 
                                  application_info['application_parameters'], 
                                  application_info['training']['group_parameters'], 
                                  variavel_de_saida, 
                                  model(**best_params),
                                  verbose=verbose)
      model_name = training_config['models'][best_model_name]['name']
      preditor_file_name = configs_file_path / f"{application_info["name"]}_{model_name}_{variavel_de_saida}.pickle"
      if verbose:
        print(f'--> Saving predictor trained with with model {best_model_name} (named {model_name}) in file {preditor_file_name}')
      predictor.save_predictor(preditor_file_name)
    else:  
        print(f"Warning: Ignoring invalid application {application_key}!")

def execute_commands(command, applications_configs, training_config, system_config, verbose):
 commands_dict = {
   'applications': partial(applications_command, applications_configs),
   'models': partial(models_command, training_config),
   'train': partial(train_command, command[1:], applications_configs, training_config, Path(system_config['predictors_path']), 
                          system_config, verbose=verbose)  
 } 

 if len(command) > 0 and command[0] in commands_dict.keys():
   commands_dict[command[0]]()
   return True
 else:  
  print("Invalid {command[0]} command!")
  return False
    
# Lê as configuraçoes;
configs_file_path, applications_configs, training_config, system_config  = read_configs()

# Processa os parâmetros do script.
command, verbose, parser = process_script_args()

# Executa os comandos do script.

Status = execute_commands(command, applications_configs, training_config, system_config, verbose)

if not Status:
  parser.print_help()
  sys.exit(-1)