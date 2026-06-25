import json
from pathlib import Path

class ReadSystemConfig:
  def __init__(self):
    self.system_config_path = None  
    self.system_config = None

  def check_system_json(self, json_data):
    return True

  def read_system_config(self, system_config_path):
    self.system_config_path = system_config_path
    with open(system_config_path, 'r') as file:
      self.system_config = json.load(file)
    if self.check_system_json(self.system_config):
      return self.system_config
    else:
      print(f"Invalid system config file {system_config_path.name}")
      return None

class ReadTrainingConfig:
  def __init__(self):
    self.training_config_path = None  
    self.training_config = None

  def check_training_json(self, json_data):
    return True

  def read_training_config(self, training_config_path):
    self.training_config_path = training_config_path
    with open(training_config_path, 'r') as file:
      self.training_config = json.load(file)
    if self.check_training_json(self.training_config):
      return self.training_config
    else:
      print(f"Invalid training config file {training_config_path.name}")
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

    for file in Path(applications_config_dir).rglob("*.json"):
        if file.is_file():
          with open(file, 'r') as json_file:
            app_json = json.load(json_file)
            if self.check_apprincation_json(app_json):
              self.applications_config[app_json['name']] = app_json
            else:
              print(f"Warning! Ignoring invalid json file {file.name} with json extension")  

        else:
          print(f"Warning! Ignoring non system file {file.name} with json extension")  
    return self.applications_config
      
class ReadUserConfig:
  def __init__(self):
    self.user_config_path = None  
    self.user_config = None

  def check_user_json(self, json_data):
    return True

  def read_user_config(self, user_config_path):
    self.user_config_path = user_config_path
    with open(user_config_path, 'r') as file:
      self.user_config = json.load(file)
    if self.check_user_json(self.user_config):
      return self.user_config
    else:
      print(f"Invalid user config file {user_config_path.name}")
      return None

class PredictorsInfoConfig:
  def __init__(self):
    self.predictors_info_config_path = None  
    self.predictors_info_config = None

  def check_predictors_info_json(self, json_data):
    return True

  def read_predictors_info_config(self, preditors_info_config_path):
    self.predictors_info_config_path = preditors_info_config_path
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
    
  def save_predictors_info_config(self, predictors_info_config):
      self.predictors_info_config = predictors_info_config
      with open(self.predictors_info_config_path, 'w') as file:
        json.dump(self.predictors_info_config, file, indent="\t")
    