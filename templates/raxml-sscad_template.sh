#!/bin/bash
#SBATCH --nodes=<<number_of_nodes>>
#SBATCH --ntasks-per-node=<<number_of_process_per_node>>
#SBATCH --ntasks=<<total_tasks>>
#SBATCH --cpus-per-task=<<threads_per_process>>
#SBATCH --partition=<<partition>>
#SBATCH --job-name=<<job_name>>
#SBATCH --time=<<max_time>>
#SBATCH --<<execution_type>>
#SBATCH --mem=<<max_memory>>

module load raxml

cd $SLURM_SUBMIT_DIR

ulimit -s unlimited
ulimit -c unlimited
ulimit -v unlimited

EXEC="/petrobr/app_sequana/raxml/8.2.12/bin/raxmlHPC-HYBRID-AVX"

PARAMS="-T $SLURM_CPUS_PER_TASK <<application_params>>"

if [ $SLURM_NTASKS_PER_NODE -eq 1 ]; then
  CPU_BIND="none"
else 
  CPU_BIND="socket"
fi       

srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK --cpu-bind=$CPU_BIND $EXEC $PARAMS

