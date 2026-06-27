import json
from pathlib import Path
from jsonschema import validate, ValidationError

schema_mestre = {
    "oneOf": [
        # ---------------------------------------------------------------------
        # SCHEMA 1: Modelos (com chaves dinâmicas para algoritmos)
        # ---------------------------------------------------------------------
        {
            "type": "object",
            "required": ["filter", "models"],
            "properties": {
                "filter": {"type": "object", "required": ["outlier_limit"]},
                "models": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "required": ["grid_search_parms", "fixed_params", "name", "import_path"]
                    }
                }
            }
        },
        
        # ---------------------------------------------------------------------
        # SCHEMA 2: Sistema (Hierárquico fixo: slurm, users_activity)
        # ---------------------------------------------------------------------
        {
            "type": "object",
            "required": ["collect_consumed_energy", "default_script_name", "users_activity", "application_params_separator", "slurm", "suggestions_names"],
            "properties": {
                "users_activity": {"type": "object", "required": ["enable", "users_executed_apps_file", "users_job_data_file"]},
                "slurm": {"type": "object", "required": ["submission_program", "partition", "max_time", "max_memory", "exclusive"]},
                "suggestions_names": {"type": "array", "items": {"type": "string"}}
            }
        },
        
        # ---------------------------------------------------------------------
        # SCHEMA 3: Caminhos (Plano e simples)
        # ---------------------------------------------------------------------
        {
            "type": "object",
            "required": ["predictors_path", "templates_path", "dataset_path", "predictors_info_config_filename"]
        },
        
        # ---------------------------------------------------------------------
        # SCHEMA 4: Parâmetros de Aplicação/Treinamento (O novo arquivo enviado)
        # ---------------------------------------------------------------------
        {
            "type": "object",
            "required": ["suggestions_parameters", "application_parameters", "name", "estimated_parameter", "training", "user"],
            "properties": {
                "suggestions_parameters": {"type": "array", "items": {"type": "string"}},
                "application_parameters": {"type": "array", "items": {"type": "string"}},
                "name": {"type": "string"},
                "estimated_parameter": {"type": "string"},
                
                # Validação do bloco de treinamento
                "training": {
                    "type": "object",
                    "required": ["group_parameters", "filter_parameters", "dataset_files"],
                    "properties": {
                        "group_parameters": {"type": "array", "items": {"type": "string"}},
                        "filter_parameters": {"type": "array", "items": {"type": "string"}},
                        "dataset_files": {"type": "array", "items": {"type": "string"}}
                    }
                },
                
                # Validação do bloco de usuário
                "user": {
                    "type": "object",
                    "required": ["executable_names", "script_template_name", "suggestions_map", "user_options", "conversions"],
                    "properties": {
                        "executable_names": {"type": "array", "items": {"type": "string"}},
                        "script_template_name": {"type": "string"},
                        "suggestions_map": {"type": "object"},
                        # user_options aceita chaves dinâmicas (ex: Bootstrap, Arquivo), mas valida a estrutura interna delas
                        "user_options": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "object",
                                "required": ["params", "type", "help"],
                                "properties": {
                                    "params": {"type": "array", "items": {"type": "string"}},
                                    "type": {"type": "string"},
                                    "help": {"type": "string"}
                                }
                            }
                        },
                        "conversions": {"type": "object"}
                    }
                }
            }
        }
    ]
}


class ReadSystemConfig:
  def __init__(self):
    self.system_config_path = None  
    self.system_config = None

  def read_system_config(self, system_config_path):
    self.system_config_path = system_config_path
    try:
      with open(system_config_path, 'r') as file:
        self.system_config = json.load(file)
 
      validate(instance=self.system_config, schema=schema_mestre)
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
  def __init__(self):
    self.training_config_path = None  
    self.training_config = None

  def read_training_config(self, training_config_path):
    self.training_config_path = training_config_path
    try:
      with open(training_config_path, 'r') as file:
        self.training_config = json.load(file)

      validate(instance=self.training_config, schema=schema_mestre)
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

            validate(instance=app_json, schema=schema_mestre)
            self.applications_config[app_json['name']] = app_json
            print(f"✅ Sucess: File {file.name} is a valid training config file!")

          else:
            print(f"Warning! Ignoring invalid json file {file.name} with json extension")  

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

      validate(instance=self.user_config, schema=schema_mestre)
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
  def __init__(self):
    self.predictors_info_config_path = None  
    self.predictors_info_config = None

  def check_predictors_info_json(self, json_data):
    return True

  def read_predictors_info_config(self, preditors_info_config_path):
    self.predictors_info_config_path = preditors_info_config_path
    try:
      if preditors_info_config_path.is_file():
        with open(preditors_info_config_path, 'r') as file:
          self.predictors_info_config = json.load(file)
        if self.check_predictors_info_json(self.predictors_info_config):
          return self.predictors_info_config
        else:
          print(f"Invalid user config file {preditors_info_config_path.name}")
          return None
      else:  
        self.predictors_info_config = {}
        return self.predictors_info_config
    except json.JSONDecodeError as e:
        print(f"Syntax Error: Invalid JSON structure.")
        print(f"Details: {e.msg} at line {e.lineno}, column {e.colno}")
        return None
    except FileNotFoundError:
        print("Error: File {training_config_path} not found.")
        return None
    
  def save_predictors_info_config(self, predictors_info_config):
      self.predictors_info_config = predictors_info_config
      with open(self.predictors_info_config_path, 'w') as file:
        json.dump(self.predictors_info_config, file, indent="\t")
    