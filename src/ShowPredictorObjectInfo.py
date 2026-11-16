import sys
from pathlib import Path
from Utils.Suggestions import SuggestionsPredictor

def show_predictor_info(preditor_file_name):
  predictor = SuggestionsPredictor.load_predictor(preditor_file_name)
  if predictor is not None:
    predictor.get_object_info()
    return True
  else:
    print(f"❌ Erro ao ler o preditor do caminho {preditor_file_name}!")  
    return False

if len(sys.argv) != 2:
  print(f"Sintaxe: {sys.argv[0]} <caminho_arquivo_preditor>")
  exit(-1)

preditor_file_name = Path(sys.argv[1])

status = show_predictor_info(preditor_file_name)

if not status:
  exit(-2)
