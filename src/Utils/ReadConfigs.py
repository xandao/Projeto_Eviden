import json
from pathlib import Path
from jsonschema import validate, ValidationError
class ReadSystemConfig:
  """
  Classe para ler as configurações do sistema, referentes aos principais
  diretórios usados e também configurações comuns aos scripts de
  treinamento e de sugestão.

  Atributos:
    system_config_path (Path | None): Caminho completo para o arquivo de
                                      configuração dos scripts de treinamentos 
                                      e do usuário.
    system_conig (dict | None): Dicionário com o arquivo de configuração 
                                convertido do formato JSON.
                                                                    
    esquema_json (dict): Esquema de validação para o script do sistema.         
    verbose (bool): Habilita/desabilita informações de verbosidade.                   
  """
  
  # Esquema de validação para o arquivo de configuração do sistema.
  esquema_json = {
    "$schema": "https://json-schema.org",
    "type": "object",
    "required": [
        "predictors_path", 
        "templates_path", 
        "dataset_path", 
        "applications_path",
        "predictors_info_config_filename"
    ],
    "properties": {
        "predictors_path": {"type": "string"},
        "templates_path": {"type": "string"},
        "dataset_path": {"type": "string"},
        "applications_path": {"type": "string"},
        "predictors_info_config_filename": {"type": "string"}
    }
  }

  def __init__(self, verbose=False):
    """
    Função de inicialização da classe ReadSystemConfig.

    Parâmetros:
      verbose (bool): Habilita/desabilita informações de verbosidade.      
    """

    # Inicializa as variáveis internas da classe.
    self.system_config_path = None  
    self.system_config = None
    self.verbose = verbose

  def read_system_config(self, system_config_path):
    """
    Função para ler o arquivo de configuração do sistema a partir do arquivo
    passado como parâmetro.

    Parâmetros:
      system_config_path (Path): Caminho completo do arquivo de configuração
                                 com a configuração do sistema em JSON.

    Retorna:
      dict | None: O dicionário com o arquivo JSON da configuração do sistema 
                   convertido para um dicionário, ou None se algum erro 
                   ocorreu ao ler ou verificar a sintaxe do arquivo ou a 
                   conformidade do JSON do arquivo com o esquema esquema_json.                                     
    """

    # Armazena o camainho do arquivo de configuração do sistema.
    self.system_config_path = system_config_path
    
    # Tenta abrir o arquivo e processá-lo se o caminho existir e for um 
    # arquivo.
    try:
      # Tenta converter o JSON, verificando a sintaxe do formato, e salva o 
      # JSON convertido para dicionário em self.system_config.
      with open(system_config_path, 'r') as file:
        self.system_config = json.load(file)
    
      # Verifica se o arquivo JSON, com a sintaxe correta, está em conformidade
      # com o esquema esquema_json.
      validate(instance=self.system_config, 
               schema=ReadSystemConfig.esquema_json)

      # Imprime a informação de sucesso se a verbosidade estiver habilitada.
      if self.verbose:
        print(f"✅ Arquivo {system_config_path.name} é um arquivo válido de",
               "configuração do sistema e foi carregado com sucesso!")

      # Retorna a configuração do sistema lida e e convertida para um 
      # dicionário.  
      return self.system_config 
        
    except ValidationError as e:
      # Ocorreu um erro ao validar o esquema do arquivo JSON lido
      print("❌ Erro ao validar o JSON do arquivo de sistema",
            f"{system_config_path.name}!")
      print(f"❌ Detalhes do erro: {e.message}")
      print(f"❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except json.JSONDecodeError as e:
      # Ocorreu um erro de sintaxe ao ler o arquivo JSON.
      print(f"❌ O arquivo {system_config_path.name} não pode ser lido como um",
            "arquivo JSON válido! Erro de sintaxe!")
      print(f"❌ Detalhes do erro: {e.msg} na linha {e.lineno} e coluna",
            f"{e.colno}!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except FileNotFoundError:
      # O arquivo não foi encontrado.
      print(f"❌ O arquivo {system_config_path.name} não foi encontrado!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except PermissionError as e:
      # O usuário que executou o script não tem permissão para acessar o 
      # arquivo.
      print("❌ Erro de permissão ao acessar o arquivo "
            f"{system_config_path.name}!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except IOError as e:
      # Ocorreu um erro de I/O ao acessar o arquivo.
      print(f"❌ Erro de I/O ao ler o arquivo {system_config_path.name}!")
      print(f"❌ Código do erro: {e.errno}; Mensagem: {e.strerror}!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except Exception as e:
      # Ocorreu alguma outra exceção, inesperada.
      print(f"❌ Erro desconhecido ao processar o arquivo",
            f"{system_config_path.name}!")
      print(f"❌ Parâmetros do erro: {e.args}!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
class ReadTrainingConfig:
  """
  Classe para ler as configurações de treinamento, usada pelo script de 
  treinamento ao treinar cada modelo. Tem as configurações de cada modelo de 
  aprendizado de máquina considerado, incluindo os hiperparâmetros que 
  otimizaremos, parâmetros fixos (por exemplo, randomstate=42) e outras 
  informações como o nome do modelo e como ele é acessado no python (para o 
  import dinâmico).

  Atributos:
    training_config_path (Path | None): Caminho completo para o arquivo de
                                        configuração dos modelos a serem 
                                        treinados.
    training_config (dict | None): Dicionário com o arquivo de configuração de
                                   treinamento convertido do formato JSON.
    esquema_json (dict): Esquema de validação para o script de treinamento.                            
    verbose (bool): Habilita/desabilita informações de verbosidade.                   
  """
  
  # Esquema de validação para o script com as configurações usadas em cada 
  # treinamento.
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
                "required": ["grid_search_parms", "fixed_params", "name", 
                             "import_path"],
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

  def __init__(self, verbose=False):
    """
    Função de inicialização da classe ReadTrainingConfig.

    Parâmetros:
      verbose (bool): Habilita/desabilita informações de verbosidade.
    """

    # Inicializa as variáveis internas da classe.
    self.training_config_path = None 
    self.training_config = None
    self.verbose = verbose

  def read_training_config(self, training_config_path):
    """
    Função para ler o arquivo de configuração com as configurações
    do treinamento dos modelos, usado pelo script de treinamento.

    Parâmetros:
      training_config_path (Path): Caminho completo do arquivo de configuração
                                   do script do usuário.

    Retorna:
      dict | None: O dicionário com o arquivo JSON da configuração do 
                   treinamento convertido para um dicionário, ou None se algum 
                   erro ocorreu ao ler ou varificar a sintaxe do arquvo.                                     
    """

    # Armazena o camainho do arquivo de configuração do treinamento,
    self.training_config_path = training_config_path

    # Tenta abrir o arquivo e processá-lo se o caminho existir e for um 
    # arquivo.
    try:
      # Tenta converter o JSON, verificando a sintaxe do formato, e salva o 
      # JSON convertido para dicionário em self.training_config.
      with open(training_config_path, 'r') as file:
        self.training_config = json.load(file)

      # Verifica se o arquivo JSON, com a sintaxe correta, está em conformidade
      # com o esquema esquema_json.
      validate(instance=self.training_config, 
               schema=ReadTrainingConfig.esquema_json)

      # Imprime a informação de sucesso se a verbosidade estiver habilitada.
      if self.verbose:
        print(f"✅ Arquivo {training_config_path.name} é um arquivo válido de",
               "configuração de treinamento e foi carregado com sucesso!")

      # Retorna a configuração do treinamento lida e convertida para um 
      # dicionário.  
      return self.training_config 
        
    except ValidationError as e:
      # Ocorreu um erro ao validar o esquema do arquivo JSON lido
      print("❌ Erro ao validar o JSON do arquivo de treinamento",
            f"{training_config_path.name}!")
      print(f"❌ Detalhes do erro: {e.message}")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except json.JSONDecodeError as e:
      # Ocorreu um erro de sintaxe ao ler o arquivo JSON.
      print(f"❌ O arquivo {training_config_path.name} não pode ser lido como",
            "um arquivo JSON válido! Erro de sintaxe!")
      print(f"❌ Detalhes do erro: {e.msg} na linha {e.lineno} e coluna",
            f"{e.colno}!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except FileNotFoundError:
      # O arquivo não foi encontrado.
      print(f"❌ O arquivo {training_config_path.name} não foi encontrado!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except PermissionError as e:
      # O usuário que executou o script não tem permissão para acessar o 
      # arquivo.
      print("❌ Erro de permissão ao acessar o arquivo",
            f"{training_config_path.name}!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except IOError as e:
      # Ocorreu um erro de I/O ao acessar o arquivo.
      print(f"❌ Erro de I/O ao ler o arquivo {training_config_path.name}!")
      print(f"❌ Código do erro: {e.errno}; Mensagem: {e.strerror}!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except Exception as e:
      # Ocorreu alguma outra exceção, inesperada.
      print("❌ Erro desconhecido ao processar o arquivo",
            f"{training_config_path.name}!")
      print(f"❌ Parâmetros do erro: {e.args}!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None

class ReadApplicationsConfigs:
  """
  Classe para ler as configurações ddos aplicativos, usada pelo script de 
  treinamento para poder treinar os modelos para o aplicativo selecionado, e 
  para o script do uruário poder fazer as sugestões. Existe um arquivo de 
  configuração para cada aplicativo, a classe lê todos os arquivos JSON do 
  diretório passado. Para cada aplicação, existem informações como os nomes das 
  variáveis de sugestão, da aplicação e do grupo usado ao detaerminar os 
  melhores hiperparâmetros para cada modelo, o melhor modelo usando, para cada 
  modelo, os melhores hiperparâmetros. Também tem as informações dos nomes dos 
  parâmetros da aplicação passados pelo usuário que são necessários para fazer 
  as sugestões e outras informações, como por exemplo, um nome para cada 
  aplicação.
  
  Atributos:
    applications_config_dir (Path | None): Caminho completo para o diretório 
                                           com os arquvios de configuração das 
                                           aplicações.
    applications_config (dict | None): Dicionário com o arquivo de configuração 
                                       de cada aplicação, usando para cada 
                                       aplicação o seu nome como chave.
    esquema_json (dict): Esquema de validação para um script de uma das 
                         aplicações.                            
    verbose (bool): Habilita/desabilita informações de verbosidade.                   
  """
  
  # Esquema de validação para o script de uma aplicação.
  esquema_json = {
    "$schema": "https://json-schema.org",
    "type": "object",
    "required": ["suggestions_parameters", "application_parameters", "name", 
                 "estimated_parameters", "training", "user"],
    "properties": {
        "suggestions_parameters": {"type": "array", 
                                   "items": {"type": "string"}},
        "application_parameters": {"type": "array", 
                                   "items": {"type": "string"}},
        "name": {"type": "string"},
        "estimated_parameters": {
          "type": "object",
          "required": ["suggestion"],
          "properties": {
            "suggestion": {"type": "string"},
            "time": {"type": "string"},
          }
        },
        
        "training": {
            "type": "object",
            "required": ["group_parameters", "filter_parameters", 
                         "dataset_files"],
            "properties": {
                "group_parameters": {"type": "array", 
                                     "items": {"type": "string"}},
                "filter_parameters": {"type": "array", 
                                      "items": {"type": "string"}},
                "dataset_files": {"type": "array", 
                                  "items": {"type": "string"}}
            }
        },
        
        "user": {
            "type": "object",
            "required": ["executable_names", "script_template_name", 
                         "suggestions_map", "user_options", "conversions"],
            "properties": {
                "executable_names": {"type": "array", 
                                     "items": {"type": "string"}},
                "script_template_name": {"type": "string"},
                "suggestions_map": {"type": "object"},
                "user_options": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "required": ["params", "type", "help"],
                        "properties": {
                            "params": {"type": "array", 
                                       "items": {"type": "string"}},
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
                        "required": ["partition", "max_time", 
                                     "max_memory", "exclusive", "default", 
                                     "nodes", "process", "threads"],
                        "properties": {
                            "partition": {"type": "string"},
                            "max_time": {"type": "integer"},    
                            "max_memory": {"type": "integer"},   
                            "exclusive": {"type": "boolean"},
                            "default": {"type": "boolean"},
                            "nodes": {"type": "integer"},    
                            "process": {"type": "integer"},    
                            "threads": {"type": "integer"},    
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

  def __init__(self, verbose=False):
    """
    Função de inicialização da classe ReadApplicationsConfigs.

    Parâmetros:
      verbose (bool): Habilita/desabilita informações de verbosidade.
    """

    # Inicializa as variáveis internas da classe.
    self.applications_config_dir = None  
    self.applications_config = None
    self.verbose = verbose

  def read_applications_config(self, applications_config_dir):
    """
    Função para ler o arquivo de configuração com as configurações do 
    treinamento de cada aplicação para a qual treinamos o melhor modelo e 
    fazemos as sugestões para o usuãrio.

    Parâmetros:
      applications_config_dir (Path): Caminho completo do diretório com os 
                                      arquivos de configurações de cada 
                                      aplicação que coletamos dados e que 
                                      podemos fazer sugestões aos usuários.                                    
    Retorna:
      dict | None: O dicionário com, para cada aplicação referenciada pelo seu 
                   nome, o arquivo JSON desta aplicação convertido para um 
                   dicionário, ou None se algum erro ocorreu ao ler ou 
                   verificar a sintaxe do arquvo.                                    
    """

    # Armazena o caminho para o diretório com os arquivos de configuração das 
    # aplicações.
    self.applications_config_dir = applications_config_dir  

    # Busca todos os arquivos no diretório com a extensão .jsom. Os scripts de 
    # treinamento e de sugestão supõe que cada arquivo no formato JSON do 
    # diretório applications_config_dir é o arquivo de configuração de uma 
    # aplicação diferente que pode ser otimizda.
    applications_files = list(Path(applications_config_dir).rglob("*.json"))

    # Se existirem arquivos de configuração a serem lidos e verificados.
    if applications_files:
      # Inicializa o dicionário com as configurações das aplicações com um 
      # dicionário vazio
      self.applications_config = {}
      # Tenta processar cada um dos arquivos no diretório 
      # applications_config_dir.
      try:
        # Para cada possível arquivo no diretório applications_config_dir.
        for file in applications_files:
            # Se file realmente for um arquivo, lemos o seu conteúdo, 
            # verificando se é um JSON válido e depois verificamos a validade 
            # do arquido de acordo com o esquema.
            if file.is_file():
              with open(file, 'r') as json_file:
                app_json = json.load(json_file)

              # Verifica se o arquivo JSON, com a sintaxe correta, está em 
              # conformidade com o esquema esquema_json.
              validate(instance=app_json, 
                       schema=ReadApplicationsConfigs.esquema_json)

              # Salva o arquivo JSON da aplicação convertido par um dicionário 
              # (em app_json) no dicionário self.applications_config, usando 
              # como chave o nome da aplicação dado na chave 'name' do 
              # dicionário app_json com a configuração da aplicação.
              self.applications_config[app_json['name']] = app_json

              # Imprime a informação de sucesso se a verbosidade estiver 
              # habilitada.
              if self.verbose:
                print(f"✅ Arquivo {file.name} é um arquivo válido de",
                       "configuração de uma aplicação e foi carregado com",
                       "sucesso!")
            else:
              # Se file não for um arquivo, mostra uma mensagem de aviso 
              # informando sobre este erro (file sempre deveria ser um arquivo).
              print(f"⚠️ Ignorando o caminho {file.name} que não é um arquivo",
                    "válido JSON!")  
              print("⚠️ Por favor, reporte este aviso ao adminstrador do",
                    "sistema!")

        # Imprime a informação de sucesso se a verbosidade estiver habilitada.    
        if self.verbose:
          print(f"✅ Todos os arquivos {applications_config_dir.name} do",
                 "diretório com as confugurações das aplicações são válidos e",
                 "foram lidos!")

        # Retorna as configurações das aplicações lidas e econvertidas para um 
        # dicionário, em que cada chaveé um dicionário com as conigurações
        #  convertidas para a aplicação identificada pela chave.
        return self.applications_config 
        
      except ValidationError as e:
        # Ocorreu um erro ao validar o esquema do arquivo JSON lido
        print("❌ Erro ao validar o JSON do arquivo da aplicação",
              f"{file.name}!")
        print(f"❌ Detalhes do erro: {e.message}")
        print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
        return None
      except json.JSONDecodeError as e:
        # Ocorreu um erro de sintaxe ao ler o arquivo JSON.
        print(f"❌ O arquivo {file.name} não pode ser lido como um arquivo",
              "JSON válido! Erro de sintaxe!")
        print(f"❌ Detalhes do erro: {e.msg} na linha {e.lineno} e coluna",
              f"{e.colno}!")
        print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
        return None
      except FileNotFoundError:
        # O arquivo não foi encontrado.
        print(f"❌ O arquivo {file.name} não foi encontrado!")
        print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
        return None
      except PermissionError as e:
        # O usuário que executou o script não tem permissão para acessar o 
        # arquivo.
        print(f"❌ Erro de permissão ao acessar o arquivo {file.name}!")
        print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
        return None
      except IOError as e:
        # Ocorreu um erro de I/O ao acessar o arquivo.
        print(f"❌ Erro de I/O ao ler o arquivo {file.name}!")
        print(f"❌ Código do erro: {e.errno}; Mensagem: {e.strerror}!")
        print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
        return None
      except Exception as e:
        # Ocorreu alguma outra exceção, inesperada.
        print(f"❌ Erro desconhecido ao processar o arquivo {file.name}!")
        print(f"❌ Parâmetros do erro: {e.args}!")
        print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
        return None
    else:
      # Verifica os erros relacionados ao diretório applications_config_dir.
      if applications_config_dir.is_dir():
        # Se for um diretório, então está vazio ou não tem arquivo com a 
        # extensão .json.
        print(f"❌ O diretório {applications_config_dir} está vazio ou não",
              "tem arquivos no formato JSON!")
        print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      else:  
        # Se não for um diretório, então applications_config_dir não é um 
        # caminho válido (por exemplo, pode ser um arquivo).
        if applications_config_dir.is_file():
          print(f"❌ O caminho {applications_config_dir} é de um arquivo e",
                "não de um diretório!")
        else:  
          print(f"❌ O caminho {applications_config_dir} não é de um",
                "diretório ou não existe!")
        print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
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
    verbose (bool): Habilita/desabilita informações de verbosidade.                   
  """

  # Esquema de validação para a configuração do script de otimização usado pelo
  # usuário do sistema.
  esquema_json = {
    "$schema": "https://json-schema.org",
    "type": "object",
    "required": [
        "collect_consumed_energy", 
        "default_script_name", 
        "users_activity", 
        "slurm" 
    ],
    "properties": {
        "collect_consumed_energy": {"type": "boolean"}, 
        "default_script_name": {"type": "string"},
        "suggestions_names": {
            "type": "array", 
            "items": {"type": "string"}
        },
        "users_activity": {
            "type": "object", 
            "required": ["enable", "data_file_prefix", "data_file_dir"],
            "properties": {
                "enable": {"type": "boolean"},
                "data_file_prefix": {"type": "string"},
                "data_file_dir": {"type": "string"},
            }
        },
        "slurm": {
            "type": "object", 
            "required": ["submission_program", "submission_message"],
            "properties": {
                "submission_program": {"type": "string"},
                "submission_message": {"type": "string"}
            }
        }
    }
  }  
  
  def __init__(self, verbose=False):
    """
    Função de inicialização da classe ReadUserConfig.

    Parâmetros:
      verbose (bool): Habilita/desabilita informações de verbosidade.
    """

    # Inicializa as variáveis internas da classe.
    self.user_config_path = None  
    self.user_config = None
    self.verbose = verbose

  def read_user_config(self, user_config_path):
    """
    Função para ler o arquivo de configuração do script do usuário do arquivo 
    passado como parâmetro.

    Parâmetros:
      user_config_path (Path): Caminho completo do arquivo de configuração do
                               script do usuário.

    Retorna:
      dict | None: O dicionário com o arquivo JSON da configuração do usuário 
                   convertido para um dicionário, ou None se algum erro ocorreu 
                   ao ler ou verificar a sintaxe do arquvo.                                     
    """

    # Armazena o camainho do arquivo de configuração do usuário.
    self.user_config_path = user_config_path
    try:
      # Tenta converter o JSON, verificando a sintaxe do formato, e salva o 
      # JSON convertido para dicionário em self.user_config.
      with open(user_config_path, 'r') as file:
        self.user_config = json.load(file)

      # Verifica se o arquivo JSON, com a sintaxe correta, está em conformidade
      # com o esquema esquema_json.
      validate(instance=self.user_config, schema=ReadUserConfig.esquema_json)

      # Imprime a informação de sucesso se a verbosidade estiver habilitada.
      if self.verbose:
        print(f"✅ Arquivo {user_config_path.name} é um arquivo válido de", 
              "configuração do script do usuário e foi carregado com sucesso!")

      # Retorna a configuração do usuário lida e e convertida para um 
      # dicionário.  
      return self.user_config 
        
    except ValidationError as e:
      # Ocorreu um erro ao validar o esquema do arquivo JSON lido
      print("❌ Erro ao validar o JSON do arquivo do script do usuário", 
            f"{user_config_path.name}!")
      print(f"❌ Detalhes do erro: {e.message}")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except json.JSONDecodeError as e:
      # Ocorreu um erro de sintaxe ao ler o arquivo JSON.
      print(f"❌ O arquivo {user_config_path.name} não pode ser lido como um",
            "arquivo JSON válido! Erro de sintaxe!")
      print(f"❌ Detalhes do erro: {e.msg} na linha {e.lineno} e coluna",
            f"{e.colno}!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except FileNotFoundError:
      # O arquivo não foi encontrado.
      print(f"❌ O arquivo {user_config_path.name} não foi encontrado!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except PermissionError as e:
      # O usuário que executou o script não tem permissão para acessar o arquivo.
      print(f"❌ Erro de permissão ao acessar o arquivo",
            f"{user_config_path.name}!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except IOError as e:
      # Ocorreu um erro de I/O ao acessar o arquivo.
      print(f"❌ Erro de I/O ao ler o arquivo {user_config_path.name}!")
      print(f"❌ Código do erro: {e.errno}; Mensagem: {e.strerror}!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except Exception as e:
      # Ocorreu alguma outra exceção, inesperada.
      print(f"❌ Erro desconhecido ao processar o arquivo", 
            f"{user_config_path.name}!")
      print(f"❌ Parâmetros do erro: {e.args}!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
class PredictorsInfoConfig:
  """
  Classe para ler as configurações que mapeiam cada apluicação ao seu preditor.

  Atributos:
    predictors_info_config_path (Path | None): Caminho completo para o arquivo 
                                               de configuração que associam 
                                               cada aplicação ao seu preditor.
    predictors_info_config (dict | None): Dicionário com o arquivo de 
                                          configuração do script com o 
                                          mapeamento de cada aplicação ao seu 
                                          preditor.
                                                                    
    esquema_json (dict): Esquema de validação para o .                            
    verbose (bool): Habilita/desabilita informações de verbosidade.                   
  """

  # Esquema de validação para as informações sobre os  preditores para cada
  # modelo treinado.
  esquema_json = {
      "type": "object",
      "patternProperties": {
          ".*": {
              "type": "string"
          }
      },
      "additionalProperties": False
  }

  def __init__(self, verbose=False):
    """
    Função de inicialização da classe PredictorsInfoConfig.

    Parâmetros:
      verbose (bool): Habilita/desabilita informações de verbosidade.
    """

    # Inicializa as variáveis internas da classe.
    self.predictors_info_config_path = None  
    self.predictors_info_config = None
    self.verbose = verbose

  def read_predictors_info_config(self, predictors_info_config_path):
    """
    Função para ler o arquivo de configuração do script que mapeia cada
    aplicação ao seu preditor.

    Parâmetros:
      predictors_info_config_path (Path): Caminho completo do arquivo de 
                                          configuração com os mapeamentos.

    Retorna:
      dict | None: O dicionário com o arquivo JSON da configuração com os 
                   mapeamentos convertido para um dicionário, ou None se algum 
                   erro ocorreu ao ler ou varificar a sintaxe do arquvo.                                     
    """

    # Armazena o camainho do arquivo de cinfiguração dos preditores..
    self.predictors_info_config_path = predictors_info_config_path

    # Tenta abrir o arquivo e processá-lo se o caminho existir e for um 
    # arquivo.
    try:
      # Tenta converter o JSON, verificando a sintaxe do formato, e salva o 
      # JSON convertido para dicionário em self.predictors_info_config.
      if predictors_info_config_path.is_file():
        with open(predictors_info_config_path, 'r') as file:
          self.predictors_info_config = json.load(file)

        # Verifica se o arquivo JSON, com a sintaxe correta, está em 
        # conformidade com o esquema esquema_json.
        validate(instance=self.predictors_info_config, 
                 schema=PredictorsInfoConfig.esquema_json)

        # Imprime a informação de sucesso se a verbosidade estiver habilitada.
        if self.verbose:
          print(f"✅ Arquivo {predictors_info_config_path.name} é um arquivo", 
                "válido de configuração dos preditores e foi carregado com", 
                "sucesso!")
      else:  
        # O arquivo somente pode não existir se ainda não treinamos nenhum 
        # modelo para nenhuma aplicação, ou seja, somente pode ocorrer no 
        # script de treinamento. No script de otimização, o arquivo sempre deve
        # existir.
        if self.verbose:
          print(f"⚠️  O arquivo {predictors_info_config_path.name} não foi",
                "encontrado, mas não tem problema se o erro for gerado pelo",
                "script de treinamento!")

        # Inicializa o dicionário de configurações como vazio, pois no 
        # primeiro treinamento não teremos ainda um preditor para uma 
        # aplicação.
        self.predictors_info_config = {}

      # Retorna a configuração dos preditores lida e e convertida para um 
      # dicionário.  
      return self.predictors_info_config

    except ValidationError as e:
      # Ocorreu um erro ao validar o esquema do arquivo JSON lido
      print("❌ Erro ao validar o JSON do arquivo do script dos preditores", 
            f"{predictors_info_config_path.name}!")
      print(f"❌ Detalhes do erro: {e.message}")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except json.JSONDecodeError as e:
      # Ocorreu um erro de sintaxe ao ler o arquivo JSON.
      print(f"❌ O arquivo {predictors_info_config_path.name} não pode ser", 
            "lido como um arquivo JSON válido! Erro de sintaxe!")
      print(f"❌ Detalhes do erro: {e.msg} na linha {e.lineno} e coluna",
            f"{e.colno}!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except FileNotFoundError:
      # O arquivo não foi encontrado.
      print(f"❌ O arquivo {predictors_info_config_path.name} não foi",
            "encontrado!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except PermissionError as e:
      # O usuário que executou o script não tem permissão para acessar o 
      # arquivo.
      print("❌ Erro de permissão ao acessar o arquivo", 
            f"{predictors_info_config_path.name}!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except IOError as e:
      # Ocorreu um erro de I/O ao acessar o arquivo.
      print("❌ Erro de I/O ao ler o arquivo",
            f"{predictors_info_config_path.name}!")
      print(f"❌ Código do erro: {e.errno}; Mensagem: {e.strerror}!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    except Exception as e:
      # Ocorreu alguma outra exceção, inesperada.
      print("❌ Erro desconhecido ao processar o arquivo", 
            f"{predictors_info_config_path.name}!")
      print(f"❌ Parâmetros do erro: {e.args}!")
      print("❌ Por favor, reporte este erro ao adminstrador do sistema!")
      return None
    
  def save_predictors_info_config(self, predictors_info_config): 
    """
    Função para salvar o arquivo de configuração do mapeamento das aplicações,
    comvertendo o dicionário com o mapeamento para o arquivo correspondente no
    formato JSON
  
    Parâmetros:
      predictors_info_config_path (Path): Caminho completo do arquivo de 
                                          configuração com os mapeamentos
                                          atualizada.

    Retorna:
      Sem retorno.                 
    """
      
    # TODO: Será que devemos copiar o arquivo anterior para um arquivo de
    #       backup altes de atualizar o arquivo (quando o arquivo já existia)
    #       para uma maior confiabilidade?
    
    # Verifica se self.predictors_info_config_path foi inicializado, ou seja,
    # se a função read_predictors_info_config foi chamada.
    if self.predictors_info_config_path is None:
      print(f"⚠️ Não foi lida a configuração dos preditores! O arquivo não", 
            "será salvo!")
    else:
      # Atualiza o dicionário das configurações dos preditores com a nova 
      # versão das configurações.
      self.predictors_info_config = predictors_info_config

      # Converte o dicionário para JSON e atualiza o arquivo originalmente lido
      # pela função self.predictors_info_config_path.
      with open(self.predictors_info_config_path, 'w') as file:
        json.dump(self.predictors_info_config, file, indent="\t")