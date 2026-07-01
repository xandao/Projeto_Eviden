import json
from pathlib import Path
from jsonschema import validate, ValidationError
class ReadSystemConfig:
  """
  Classe para ler as configurações do sistema, usada pelos scripts de 
  treinamento e do usuário.

  Atributos:
    system_config_path (Path | None): Caminho completo para o arquivo de
                                      configuração dos scripts de treinamento 
                                      e do usuário.
    system_config (dict | None): Dicionário com o arquivo de configuração dos
                                 scripts convertido do formato JSON.
                                                                    
  """
  
  # ---------------------------------------------------------------------
  # Esquema de validação para o script do sistema, usado pelo script que 
  # treina os modelos e o script usado pelo usário do sistema (tem os 
  # caminhos dos diretórios usados).
  # ---------------------------------------------------------------------
  esquema_json = {
    "$schema": "https://json-schema.org",
    "type": "object",
    "required": [
        "predictors_path", 
        "templates_path", 
        "dataset_path", 
        "predictors_info_config_filename"
    ],
    "properties": {
        "predictors_path": {"type": "string"},
        "templates_path": {"type": "string"},
        "dataset_path": {"type": "string"},
        "predictors_info_config_filename": {"type": "string"}
    }
  }
  def __init__(self):
    """
    Função de inicialização da classe ReadSystemConfig.

    Parâmetros:

      Não tem.
    """
    self.system_config_path = None  
    self.system_config = None

  def read_system_config(self, system_config_path: Path) -> dict:
    """
    Função para ler o arquivo de configuração do script do usuário do
    arquivo passado como parâmetro.

    Parâmetros:
      system_config_path (Path): Caminho completo do arquivo de configuração
                                 do script do usuário.

    Retorna:
      dict: dicionário com o arquivo JSON da configuração do usuário convertido
            para o dicionário.
                                     
    """
    self.system_config_path = system_config_path
    try:
      with open(system_config_path, 'r') as file:
        self.system_config = json.load(file)
 
      validate(instance=self.system_config, schema=ReadSystemConfig.esquema_json)
      print(f"✅ Sucess: File {system_config_path.name} is a valid system config file!")
      return self.system_config 
        
    except ValidationError as e:
    # Quando falha no 'oneOf', ele avisa que não bateu com nenhum molde
      print(f"❌ Validation error in file {system_config_path.name}!")
      print(f"   Details of error: {e.message}")
      return None
    except json.JSONDecodeError:
      print(f"❌ Critical error: File {system_config_path.name} could not be read as valid JSON (syntax error)!")
      print(f"   Details: {e.msg} at line {e.lineno}, column {e.colno}!")
      return None
    except FileNotFoundError:
      print(f"❌ Critical error: File {system_config_path.name} not found!")
      return None
class ReadTrainingConfig:
  """
  Classe para ler as configurações de treinamento, usada pelo script de 
  treinamento ao treinar cada modelo. Tem as configurações de cada 
  modelo de aprendizado de máquina considerado, incluindo os 
  hiperparâmetros que otimizamos, parâmetros fixos (por exemplo, 
  randomstate=42) e outras informação como o nome do modelo e como ele
  ẽ acessado no python (o import).

  Atributos:
    training_config_path (Path | None): Caminho completo para o arquivo de
                                        configuração do script de treinamento.
    training_config (dict | None): Dicionário com o arquivo de configuração de
                                   treinamento convertido do formato JSON.
  """
  
  # ---------------------------------------------------------------------
  # Esquema de validação para o script com as configurações usadas em 
  # cada treinamento.
  # ---------------------------------------------------------------------

  esquema_json = {
    "$schema": "https://json-schema.org",
    "type": "object",
    "required": ["filter", "models"],
    "properties": {
        "filter": {
            "type": "object", 
            "required": ["outlier_limit"],
            "properties": {
                "outlier_limit": {"type": "number"}  # Adicionado o tipo do campo
            }
        },
        "models": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["grid_search_parms", "fixed_params", "name", "import_path"],
                "properties": {
                    "grid_search_parms": {"type": "object"},  # Tipo sugerido
                    "fixed_params": {"type": "object"},       # Tipo sugerido
                    "name": {"type": "string"},               # Tipo sugerido
                    "import_path": {"type": "string"}         # Tipo sugerido
                }
            }
        }
    }
  } 

  def __init__(self):
    self.training_config_path = None  
    self.training_config = None

  def read_training_config(self, training_config_path):
    self.training_config_path = training_config_path
    try:
      with open(training_config_path, 'r') as file:
        self.training_config = json.load(file)

      validate(instance=self.training_config, schema=ReadTrainingConfig.esquema_json)
      print(f"✅ Sucess: File {training_config_path.name} is a valid training config file!")
      return self.training_config 
        
    except ValidationError as e:
    # Quando falha no 'oneOf', ele avisa que não bateu com nenhum molde
      print(f"❌ Validation error in file {training_config_path.name}!")
      print(f"   Details of error: {e.message}")
      return None
    except json.JSONDecodeError as e:
      print(f"❌ Critical error: File {training_config_path.name} could not be read as valid JSON (syntax error)!")
      print(f"   Details: {e.msg} at line {e.lineno}, column {e.colno}")
      return None
    except FileNotFoundError:
      print(f"❌ Critical error: File {training_config_path.name} not found!")
      return None

class ReadApplicationsConfigs:
  # ---------------------------------------------------------------------
  # Esquema de validação para o script de uma aplicação.
  # ---------------------------------------------------------------------
  esquema_json = {
    "$schema": "https://json-schema.org",
    "type": "object",
    "required": ["suggestions_parameters", "application_parameters", "name", "estimated_parameter", "training", "user"],
    "properties": {
        "suggestions_parameters": {"type": "array", "items": {"type": "string"}},
        "application_parameters": {"type": "array", "items": {"type": "string"}},
        "name": {"type": "string"},
        "estimated_parameter": {"type": "string"},
        
        "training": {
            "type": "object",
            "required": ["group_parameters", "filter_parameters", "dataset_files"],
            "properties": {
                "group_parameters": {"type": "array", "items": {"type": "string"}},
                "filter_parameters": {"type": "array", "items": {"type": "string"}},
                "dataset_files": {"type": "array", "items": {"type": "string"}}
            }
        },
        
        "user": {
            "type": "object",
            "required": ["executable_names", "script_template_name", "suggestions_map", "user_options", "conversions"],
            "properties": {
                "executable_names": {"type": "array", "items": {"type": "string"}},
                "script_template_name": {"type": "string"},
                "suggestions_map": {"type": "object"},
                "user_options": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "required": ["params", "type", "help"],
                        "properties": {
                            "params": {"type": "array", "items": {"type": "string"}},
                            "type": {"type": "string"},  # Campo literal do seu JSON
                            "help": {"type": "string"}
                        }
                    }
                },
                "conversions": {"type": "object"}
            }
        }
    }
  } 

  def __init__(self):
    self.applications_config_dir = None  
    self.applications_config = None

  def check_apprincation_json(self, json_data):
    return True

  def read_applications_config(self, applications_config_dir):
    self.applications_config_dir = applications_config_dir  
    self.applications_config = {}
    try:
      for file in Path(applications_config_dir).rglob("*.json"):
          if file.is_file():
            with open(file, 'r') as json_file:
              app_json = json.load(json_file)

            validate(instance=app_json, schema=ReadApplicationsConfigs.esquema_json)
            self.applications_config[app_json['name']] = app_json
            print(f"✅ Sucess: File {file.name} is a valid training config file!")

          else:
            print(f"⚠️ Warning! Ignoring invalid json file {file.name} with json extension")  

      print(f"✅ Sucess: Files in {applications_config_dir.name} applications config directory are all valid application scripts!")
      return self.applications_config 
        
    except ValidationError as e:
    # Quando falha no 'oneOf', ele avisa que não bateu com nenhum molde
      print(f"❌ Validation error in file {file.name}!")
      print(f"   Details of error: {e.message}")
      return None
    except json.JSONDecodeError as e:
      print(f"❌ Critical error: File {file.name} could not be read as valid JSON (syntax error)!")
      print(f"   Details: {e.msg} at line {e.lineno}, column {e.colno}")
      return None
    except FileNotFoundError:
      print(f"❌ Critical error: File {file.name} not found!")
      return None
  
class ReadUserConfig:
  # ---------------------------------------------------------------------
  # Esquema de validação para a configuração do script de otimização usado 
  # pelo usuário do sistema.
  # ---------------------------------------------------------------------
  esquema_json = {
    "$schema": "https://json-schema.org",
    "type": "object",
    "required": [
        "collect_consumed_energy", 
        "default_script_name", 
        "users_activity", 
        "application_params_separator", 
        "slurm", 
        "suggestions_names"
    ],
    "properties": {
        # Campos principais que faltavam no seu properties (ajuste os tipos se necessário)
        "collect_consumed_energy": {"type": "boolean"}, 
        "default_script_name": {"type": "string"},
        "application_params_separator": {"type": "string"},
        
        "suggestions_names": {
            "type": "array", 
            "items": {"type": "string"}
        },
        
        "users_activity": {
            "type": "object", 
            "required": ["enable", "users_executed_apps_file", "users_job_data_file"],
            "properties": {
                "enable": {"type": "boolean"},
                "users_executed_apps_file": {"type": "string"},
                "users_job_data_file": {"type": "string"}
            }
        },
        
        "slurm": {
            "type": "object", 
            "required": ["submission_program", "partition", "max_time", "max_memory", "exclusive"],
            "properties": {
                "submission_program": {"type": "string"},
                "partition": {"type": "string"},
                "max_time": {"type": "string"},     # Ex: "02:00:00" ou número de minutos
                "max_memory": {"type": "string"},   # Ex: "4G" ou número de MB
                "exclusive": {"type": "boolean"}
            }
        }
    }
  }  
  
  def __init__(self):
    self.user_config_path = None  
    self.user_config = None

  def check_user_json(self, json_data):
    return True

  def read_user_config(self, user_config_path):
    self.user_config_path = user_config_path
    try:
      with open(user_config_path, 'r') as file:
        self.user_config = json.load(file)

      validate(instance=self.user_config, schema=ReadUserConfig.esquema_json)
      print(f"✅ Sucess: File {user_config_path.name} is a valid applications config file!")
      return self.user_config 
        
    except ValidationError as e:
    # Quando falha no 'oneOf', ele avisa que não bateu com nenhum molde
      print(f"❌ Validation error in file {user_config_path.name}!")
      print(f"   Details of error: {e.message}")
      return None
    except json.JSONDecodeError as e:
      print(f"❌ Critical error: File {user_config_path.name} could not be read as valid JSON (syntax error)!")
      print(f"   Details: {e.msg} at line {e.lineno}, column {e.colno}")
      return None
    except FileNotFoundError:
      print(f"❌ Critical error: File {user_config_path.name} not found!")
      return None

class PredictorsInfoConfig:
  # ---------------------------------------------------------------------
  # SCHEMA 5: Esquema de validação para as informações sobre os 
  #           preditores para cada modelo treinado.
  # ---------------------------------------------------------------------
  esquema_json = {
      "type": "object",
      "patternProperties": {
          ".*": {
              "type": "string"
          }
      },
      "additionalProperties": False
  }

  def __init__(self):
    self.predictors_info_config_path = None  
    self.predictors_info_config = None

  def read_predictors_info_config(self, predictors_info_config_path):
    self.predictors_info_config_path = predictors_info_config_path
    try:
      if predictors_info_config_path.is_file():
        with open(predictors_info_config_path, 'r') as file:
          self.predictors_info_config = json.load(file)
        validate(instance=self.predictors_info_config, schema=PredictorsInfoConfig.esquema_json)
        print(f"✅ Sucess: File {predictors_info_config_path.name} is a valid predictors config file!")
      else:  
        print(f"⚠️ Warning: Path {predictors_info_config_path.name} does not exist! No problem if no model has been trained yet!")
        self.predictors_info_config = {}
      return self.predictors_info_config

    except ValidationError as e:
    # Quando falha no 'oneOf', ele avisa que não bateu com nenhum molde
      print(f"❌ Validation error in file {predictors_info_config_path.name}!")
      print(f"   Details of error: {e.message}")
      return None
    except json.JSONDecodeError as e:
      print(f"❌ Critical error: File {predictors_info_config_path.name} could not be read as valid JSON (syntax error)!")
      print(f"   Details: {e.msg} at line {e.lineno}, column {e.colno}")
      return None
    except FileNotFoundError:
      print(f"❌ Critical error: File {predictors_info_config_path.name} not found!")
      return None
    
  def save_predictors_info_config(self, predictors_info_config):
      self.predictors_info_config = predictors_info_config
      with open(self.predictors_info_config_path, 'w') as file:
        json.dump(self.predictors_info_config, file, indent="\t")
    