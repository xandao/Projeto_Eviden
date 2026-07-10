import json
from pathlib import Path
from jsonschema import validate, ValidationError
from Utils.Common import debug_code
class ReadSystemConfig:
  """
  Classe para ler as configurações do sistema, comuns aos scripts de 
  treinamento e do usuário.

  Atributos:
    system_config_path (Path | None): Caminho completo para o arquivo de
                                      configuração dos scripts de treinamento 
                                      e do usuário.
    system_config (dict | None): Dicionário com o arquivo de configuração comuns 
                                 dos scripts convertido do formato JSON.
                                                                    
    esquema_json (dict): Esquema de validação para o script do sistema.                            
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

  def read_system_config(self, system_config_path: Path) -> dict | None:
    """
    Função para ler o arquivo de configuração do script do usuário do
    arquivo passado como parâmetro.

    Parâmetros:
      system_config_path (Path): Caminho completo do arquivo de configuração
                                 do script do usuário.

    Retorna:
      dict | None: O dicionário com o arquivo JSON da configuração do usuário 
                   convertido para um dicionário, ou None se algum erro ocorreu
                   ao ler ou varificar a sintaxe do arquvo.                                     
    """
    self.system_config_path = system_config_path
    try:
      with open(system_config_path, 'r') as file:
        self.system_config = json.load(file)
 
      validate(instance=self.system_config, schema=ReadSystemConfig.esquema_json)
      if debug_code:
        print(f"✅ Sucesso: Arquivo {system_config_path.name} é um arquivo válido de configuração do sistema e foi carregado com sucesso!")
      return self.system_config 
        
    except ValidationError as e:
    # Quando falha no 'oneOf', ele avisa que não bateu com nenhum molde
      print(f"❌ Erro ao validar o JSON do arquivo de sistema {system_config_path.name}!")
      print(f"   Detalhes do erro: {e.message}")
      print(f"   Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except json.JSONDecodeError:
      print(f"❌ Erro crítico: O arquivo {system_config_path.name} não pode ser lido como um arquivo JSON válido! Erro de sintaxe!")
      print(f"   Detalhes do erro: {e.msg} at line {e.lineno}, column {e.colno}!")
      print(f"   Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except FileNotFoundError:
      print(f"❌ Erro crítico: Arquivo {system_config_path.name} não foi encontrado!")
      print(f"   Por favor, reporte este erro ao adminstrador do sistema!")
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
    esquema_json (dict): Esquema de validação para o script de treinamento.                            
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
                "outlier_limit": {"type": "number"}  
            }
        },
        "models": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "required": ["grid_search_parms", "fixed_params", "name", "import_path"],
                "properties": {
                    "grid_search_parms": {"type": "object"},  
                    "fixed_params": {"type": "object"},       
                    "name": {"type": "string"},               
                    "import_path": {"type": "string"}         
                }
            }
        }
    }
  } 

  def __init__(self):
    """
    Função de inicialização da classe ReadTrainingConfig.

    Parâmetros:

      Não tem.
    """
    self.training_config_path = None  
    self.training_config = None

  def read_training_config(self, training_config_path: Path) -> dict | None:
    """
    Função para ler o arquivo de configuração com as configurações
    do treinamento dos modelos, usado pelo script de treinamento.

    Parâmetros:
      training_config_path (Path): Caminho completo do arquivo de configuração
                                   do script do usuário.

    Retorna:
      dict | None: O dicionário com o arquivo JSON da configuração do treinamento convertido
                   para um dicionário, ou None se algum erro ocorreu ao ler ou varificar a 
                   sintaxe do arquvo.                                     
    """
    self.training_config_path = training_config_path
    try:
      with open(training_config_path, 'r') as file:
        self.training_config = json.load(file)

      validate(instance=self.training_config, schema=ReadTrainingConfig.esquema_json)
      if debug_code:
        print(f"✅ Sucesso: Arquivo {training_config_path.name} é um arquivo válido de configuração de treinamento e foi carregado com sucesso!")
      return self.training_config 
        
    except ValidationError as e:
    # Quando falha no 'oneOf', ele avisa que não bateu com nenhum molde
      print(f"❌ Erro ao validar o JSON do arquivo de treinamento {training_config_path.name}!")
      print(f"   Detalhes do erro: {e.message}")
      print(f"   Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except json.JSONDecodeError:
      print(f"❌ Erro crítico: O arquivo {training_config_path.name} não pode ser lido como um arquivo JSON válido! Erro de sintaxe!")
      print(f"   Detalhes do erro: {e.msg} at line {e.lineno}, column {e.colno}!")
      print(f"   Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except FileNotFoundError:
      print(f"❌ Erro crítico: Arquivo {training_config_path.name} não foi encontrado!")
      print(f"   Por favor, reporte este erro ao adminstrador do sistema!")
      return None

class ReadApplicationsConfigs:
  """
  Classe para ler as configurações ddos aplicativos, usada pelo script de 
  treinamento para poder treinar os modelos para o aplicativo selecionado,
  e para o script do uruário poder fazer as sugestões. Existe um arquivo
  de configuração para cada aplicativo, a classe lê todos os arquivos
  JSON do diretório passado. Para cada aplicação, existem informações
  como os nomes das variáveis de sugestão, da aplicação e do grupo
  usado ao detaerminar os melhores hiperparâmetros para cada modelo, o
  melhor modelo usando, para cada modelo, os melhores hiperparâmetros.
  Também tem as informações dos nomes dos parâmetros da aplicação passados
  pelo usuário que são necessários para fazer as sugestões e outras
  informações, como por exemplo, um nome para cada aplicação.
  
  Atributos:
    applications_config_dir (Path | None): Caminho completo para o diretório com
                                           os arquvios de configuração das 
                                           aplicações.
    applications_config (dict | None): Dicionário com o arquivo de configuração de
                                       cada aplicação, usando para cada aplicação
                                       o seu nome como chave.
    esquema_json (dict): Esquema de validação para um script de uma das 
                         aplicações.                            
  """
  
  # ---------------------------------------------------------------------
  # Esquema de validação para o script de uma aplicação.
  # ---------------------------------------------------------------------
  esquema_json = {
    "$schema": "https://json-schema.org",
    "type": "object",
    "required": ["suggestions_parameters", "application_parameters", "name", "estimated_parameters", "training", "user"],
    "properties": {
        "suggestions_parameters": {"type": "array", "items": {"type": "string"}},
        "application_parameters": {"type": "array", "items": {"type": "string"}},
        "name": {"type": "string"},
        "estimated_parameters": {
          "type": "object",
          "required": ["suggestion"],
          "properties": {
            "suggestion": {"type": "string"},
            "time": {"type": "string"},
            "memory": {"type": "string"},
          }
        },
        
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
                            "type": {"type": "string"},  
                            "help": {"type": "string"}
                        }
                    }
                },
                "conversions": {"type": "object"},
                "slurm": {
                    "type": "array",
                    "items": {
                        "type": "object", 
                        "required": ["partition", "max_time", "max_memory", "exclusive", "default"],
                        "properties": {
                            "partition": {"type": "string"},
                            "max_time": {"type": "integer"},    
                            "max_memory": {"type": "integer"},   
                            "exclusive": {"type": "boolean"},
                            "default": {"type": "boolean"}
                        },
                    },
                    "contains": {
                        "type": "object",
                        "properties": {
                          "default": { "const": True }
                        }
                    },
                    "minContains": 1,
                    "maxContains": 1
                }
            }
        }
    }
  } 

  def __init__(self):
    """
    Função de inicialização da classe ReadApplicationsConfigs.

    Parâmetros:

      Não tem.
    """
    self.applications_config_dir = None  
    self.applications_config = None

  def read_applications_config(self, applications_config_dir: Path) -> dict | None:
    """
    Função para ler o arquivo de configuração com as configurações
    do treinamento de cada aplicação para a qual treinamos o melhor 
    modelo e fazemos as sugestões para o usuãrio.

    Parâmetros:
      applications_config_dir (Path): Caminho completo do diretório com os arquivos 
                                      de configurações  de cada aplicação que coletamos
                                      dados e que podemos fazer sugestões aos usuários.                                    
    Retorna:
      dict | None: O dicionário com, para cada aplicação referenciada pelo seu nome, o arquivo 
                   JSON desta aplicação convertido para um dicionário, ou None se algum erro 
                   ocorreu ao ler ou varificar a sintaxe do arquvo.                                    
    """
    self.applications_config_dir = applications_config_dir  
    self.applications_config = {}
    try:
      for file in Path(applications_config_dir).rglob("*.json"):
          if file.is_file():
            with open(file, 'r') as json_file:
              app_json = json.load(json_file)

            validate(instance=app_json, schema=ReadApplicationsConfigs.esquema_json)
            self.applications_config[app_json['name']] = app_json
            if debug_code:
              print(f"✅ Sucesso: Arquivo {file.name} é um arquivo válido de configuração de uma aplicação e foi carregado com sucesso!")
          else:
            print(f"⚠️ Aviso: Ignorando o caminho {file.name} que não é um arquivo válido JSON!")  
            print(f"   Por favor, reporte este aviso ao adminstrador do sistema!")
  
      if debug_code:
        print(f"✅ Sucesso: Todos os arquicos {applications_config_dir.name} do diretório com as confugurações das aplicações são válidos e foram lidos!")
      return self.applications_config 
        
    except ValidationError as e:
    # Quando falha no 'oneOf', ele avisa que não bateu com nenhum molde
      print(f"❌ Erro ao validar o JSON do arquivo da aplicação {file.name}!")
      print(f"   Detalhes do erro: {e.message}")
      print(f"   Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except json.JSONDecodeError:
      print(f"❌ Erro crítico: O arquivo {file.name} não pode ser lido como um arquivo JSON válido! Erro de sintaxe!")
      print(f"   Detalhes do erro: {e.msg} at line {e.lineno}, column {e.colno}!")
      print(f"   Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except FileNotFoundError:
      print(f"❌ Erro crítico: Arquivo {file.name} não foi encontrado!")
      print(f"   Por favor, reporte este erro ao adminstrador do sistema!")
      return None
 
class ReadUserConfig:
  """
  Classe para ler as configurações usadas pelo script do usuário.

  Atributos:
    user_config_path (Path | None): Caminho completo para o arquivo de
                                    configuração do scripts do usuário.
    user_config (dict | None): Dicionário com o arquivo de configuração do
                              script do usuário convertido do formato JSON.
                                                                    
    esquema_json (dict): Esquema de validação para o script do usuário.                            
"""

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
            "required": ["enable", "users_executed_apps_file", "users_jobs_data_file"],
            "properties": {
                "enable": {"type": "boolean"},
                "users_executed_apps_file": {"type": "string"},
                "users_jobs_data_file": {"type": "string"}
            }
        },
        
        "slurm": {
            "type": "object", 
            "required": ["submission_program"],
            "properties": {
                "submission_program": {"type": "string"},
            }
        }
    }
  }  
  
  def __init__(self):
    """
    Função de inicialização da classe ReadUserConfig.

    Parâmetros:

      Não tem.
    """
    self.user_config_path = None  
    self.user_config = None

  def read_user_config(self, user_config_path: Path) -> dict | None:
    """
    Função para ler o arquivo de configuração do script do usuário do
    arquivo passado como parâmetro.

    Parâmetros:
      user_config_path (Path): Caminho completo do arquivo de configuração
                               do script do usuário.

    Retorna:
      dict | None: O dicionário com o arquivo JSON da configuração do usuário convertido
                   para um dicionário, ou None se algum erro ocorreu ao ler ou varificar 
                   a sintaxe do arquvo.                                     
    """
    self.user_config_path = user_config_path
    try:
      with open(user_config_path, 'r') as file:
        self.user_config = json.load(file)

      validate(instance=self.user_config, schema=ReadUserConfig.esquema_json)
      if debug_code:
        print(f"✅ Sucesso: Arquivo {user_config_path.name} é um arquivo válido de configuração do script do usuário e foi carregado com sucesso!")
      return self.user_config 
        
    except ValidationError as e:
    # Quando falha no 'oneOf', ele avisa que não bateu com nenhum molde
      print(f"❌ Erro ao validar o JSON do arquivo do script do usuário {user_config_path.name}!")
      print(f"   Detalhes do erro: {e.message}")
      print(f"   Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except json.JSONDecodeError:
      print(f"❌ Erro crítico: O arquivo {user_config_path.name} não pode ser lido como um arquivo JSON válido! Erro de sintaxe!")
      print(f"   Detalhes do erro: {e.msg} at line {e.lineno}, column {e.colno}!")
      print(f"   Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except FileNotFoundError:
      print(f"❌ Erro crítico: Arquivo {user_config_path.name} não foi encontrado!")
      print(f"   Por favor, reporte este erro ao adminstrador do sistema!")
      return None
class PredictorsInfoConfig:
  """
  Classe para ler as configurações que mapeiam cada apluicação ao seu 
  preditor.

  Atributos:
    predictors_info_config_path (Path | None): Caminho completo para o arquivo de
                                               configuração que associam cada 
                                               aplicação ao seu preditor.
    predictors_info_config (dict | None): Dicionário com o arquivo de configuração do
                                          script com o mapeamento de cada aplicação ao
                                          seu preditor.
                                                                    
    esquema_json (dict): Esquema de validação para o .                            
  """
  # ---------------------------------------------------------------------
  # Esquema de validação para as informações sobre os  preditores para 
  # cada modelo treinado.
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
    """
    Função de inicialização da classe PredictorsInfoConfig.

    Parâmetros:

      Não tem.
    """
    self.predictors_info_config_path = None  
    self.predictors_info_config = None

  def read_predictors_info_config(self, predictors_info_config_path: Path) -> dict | None:
    """
    Função para ler o arquivo de configuração do script que mapeia cada
    aplicação ao seu preditor.

    Parâmetros:
      predictors_info_config_path (Path): Caminho completo do arquivo de 
                                          configuração com os mapeamentos.

    Retorna:
      dict | None: O dicionário com o arquivo JSON da configuração com os mapeamentos 
                   convertido para um dicionário, ou None se algum erro ocorreu
                   ao ler ou varificar a sintaxe do arquvo.                                     
    """
    self.predictors_info_config_path = predictors_info_config_path
    try:
      if predictors_info_config_path.is_file():
        with open(predictors_info_config_path, 'r') as file:
          self.predictors_info_config = json.load(file)
        validate(instance=self.predictors_info_config, schema=PredictorsInfoConfig.esquema_json)
        if debug_code:
          print(f"✅ Sucesso: Arquivo {predictors_info_config_path.name} é um arquivo válido de configuração dos preditores e foi carregado com sucesso!")
      else:  
        if debug_code:
          print(f"⚠️ Aviso: O arquivo {predictors_info_config_path.name} não existe, mas não tem problema se o erro for gerado pelo script de treinamento!")
        self.predictors_info_config = {}
      return self.predictors_info_config

    except ValidationError as e:
    # Quando falha no 'oneOf', ele avisa que não bateu com nenhum molde
      print(f"❌ Erro ao validar o JSON do arquivo do script do usuário {predictors_info_config_path.name}!")
      print(f"   Detalhes do erro: {e.message}")
      print(f"   Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except json.JSONDecodeError:
      print(f"❌ Erro crítico: O arquivo {predictors_info_config_path.name} não pode ser lido como um arquivo JSON válido! Erro de sintaxe!")
      print(f"   Detalhes do erro: {e.msg} at line {e.lineno}, column {e.colno}!")
      print(f"   Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except FileNotFoundError:
      print(f"❌ Erro crítico: Arquivo {predictors_info_config_path.name} não foi encontrado!")
      print(f"   Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    
  def save_predictors_info_config(self, predictors_info_config: Path) -> None: 
      """
      Função para salvar o arquivo de configuração do mapeamento das aplicações,
      comvertendo o dicionário com o mapeamento para o arquivo correspondente
      no formato JSON
      """
      
      # TODO: Será que devemos copiar o arquivo anterior para um arquivo de
      #       backup altes de atualizar o arquivo (quando o arquivo já existia)
      #       para uma maior confiabilidade?

      self.predictors_info_config = predictors_info_config
      with open(self.predictors_info_config_path, 'w') as file:
        json.dump(self.predictors_info_config, file, indent="\t")
    