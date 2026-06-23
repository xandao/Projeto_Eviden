#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=24
#SBATCH --partition=sequana_cpu_dev
#SBATCH --job-name=1n-2m-24th-Ac-bt-mz
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --mem=350G

module load nas/1.0

cd $SLURM_SUBMIT_DIR

ulimit -s unlimited
ulimit -c unlimited
ulimit -v unlimited


EXEC="bt-mz.A.x"
CLASS="A"
BENCHMARK=`basename ${EXEC}`
CPU_BIND="sockets"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_DISPLAY_ENV=TRUE
export OMP_DISPLAY_AFFINITY=TRUE
export OMP_AFFINITY_FORMAT="Processo_id: %P; Thread_id %n bound to CPU %A; Host: %H; Número de Threads %N; Thread_id no sistema: %i"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMPI_MCA_ess_base_verbose=5
echo $EXEC
echo "Distribuicao de cores antes da execucao:"
lscpu | grep "Core(s) per socket"
lscpu | grep "Socket(s)"
lscpu | grep "Thread(s) per core"
lscpu | grep "Model name"
dmesg | grep rapl

echo "Executando coleta de energia: coleta feita apenas pelo processo MPI 0 para o nó inteiro"

srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK --cpu-bind=verbose,$CPU_BIND bash -c '

  NODE=$(hostname)
  JOBID=$SLURM_JOB_ID
  TASKID=$SLURM_PROCID
  LOCALID=$SLURM_LOCALID

  OUTPUT_FILE="energy_during_execution_'"$BENCHMARK"'-${SLURM_CPUS_PER_TASK}th-${SLURM_NNODES}n-${SLURM_NTASKS_PER_NODE}m-'"${CLASS}"'c-rep'"${1}"'-${NODE}.csv"

  if [ $LOCALID -eq 0 ]; then
    # Criar cabecalho do CSV apenas se o arquivo nao existir
    echo "TIMESTAMP, ENERGY_P0 (MJ), ENERGY_P1 (MJ), DRAM_P0 (MJ), DRAM_P1 (MJ), NODE, BENCHMARK, CLASS, NO, PROC P/ NO, THREAD P/ NO, REPETICAO, JOB_ID, TIME (S)" > ${OUTPUT_FILE}

    coletar_energia() {
      while true; do
        TIMESTAMP=$(date +%s)
        ENERGY_P0=$(cat /sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj)
        ENERGY_P1=$(cat /sys/class/powercap/intel-rapl/intel-rapl:1/energy_uj)
        DRAM_P0=$(cat /sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj)
        DRAM_P1=$(cat /sys/class/powercap/intel-rapl/intel-rapl:1/intel-rapl:1:0/energy_uj)
        echo "$TIMESTAMP, $ENERGY_P0, $ENERGY_P1, $DRAM_P0, $DRAM_P1, $NODE" >> "${OUTPUT_FILE}"
        sleep 1s
      done
    }

    echo "Iniciando medicao continua de energia..."
    coletar_energia &
    PID=$!
  fi

  echo "Execucao do '"$BENCHMARK"' ..."
  START=$(date +%s)  # Inicio da execucao
  '"$EXEC"'
  ret_exec=$?
  END=$(date +%s)  # Fim da execucao

  if [ $LOCALID -eq 0 ]; then
    echo "Finalizando medicao continua de energia..."
    kill $PID

    EXEC_TIME=$((END - START))

    sed -i "$ s/$/,'"${BENCHMARK}"'/" ${OUTPUT_FILE}
    sed -i "$ s/$/,'"${CLASS}"'/" ${OUTPUT_FILE}
    sed -i "$ s/$/,${SLURM_NNODES}/" ${OUTPUT_FILE}
    sed -i "$ s/$/,${SLURM_NTASKS_PER_NODE}/" ${OUTPUT_FILE}
    sed -i "$ s/$/,${SLURM_CPUS_PER_TASK}/" ${OUTPUT_FILE}
    sed -i "$ s/$/,'"${1}"'/" ${OUTPUT_FILE}
    sed -i "$ s/$/,${SLURM_JOB_ID}/" ${OUTPUT_FILE}
    sed -i "$ s/$/,${EXEC_TIME}/" ${OUTPUT_FILE}

  echo "Tempo de execucao: $EXEC_TIME segundos"
  fi

  if [ $ret_exec -ne 0 ]; then
      echo "[ERRO interno] Código $ret_exec (falha no executável $BENCHMARK)" >&2
      exit $ret_exec    # <-- propaga erro pro srun
  fi
'

### Verificação pós-execução 
ret=$?

echo -e "
Retorno srun: $ret 
"

### Registro centralizado de status
LOGGER="resumo_execucoes.csv"
DATE_NOW=$(date '+%F %T')

# Cria cabeçalho do CSV apenas se o arquivo ainda não existe
if [ ! -f "$LOGGER" ]; then
  echo "DataHora;JobID;Benchmark;Classe;Threads p/ proc; Processo p/ no; No; Retorno Srun;EstadoSLURM;Mensagem" > "$LOGGER"
fi

# Checa também o estado no SLURM (para capturar TIMEOUT, OOM etc.)
state=$(sacct -j $SLURM_JOB_ID --format=State%20 -n | tail -1 | xargs)

case $state in
  FAILED)      msg="Falha geral (SLURM: FAILED)";;
  OUT_OF_MEMORY) msg="Falha por memória (SLURM: OOM)";;
  TIMEOUT)     msg="Tempo limite excedido";;
  CANCELLED)   msg="Job cancelado";;
  COMPLETED)
      if [ $ret -ne 0 ]; then
          msg="Falha no executável (ret=$ret)"
      else
          msg="OK"
      fi
      ;;
  *) msg="Estado SLURM desconhecido: $state (ret=$ret)";;
esac

DATE_END=$(date '+%F %T')
echo "$DATE_END;$SLURM_JOB_ID;$BENCHMARK;$CLASS;$OMP_NUM_THREADS;$SLURM_NTASKS_PER_NODE;$SLURM_NNODES;$ret;$state;$msg" >> "$LOGGER"

echo "=== Fim do job $SLURM_JOB_ID $BENCHMARK.$CLASS $OMP_NUM_THREADS threads $SLURM_NTASKS_PER_NODE processo MPI por no $SLURM_NNODES no - $msg ==="

