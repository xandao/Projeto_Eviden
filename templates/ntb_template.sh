#SBATCH --nodes=<<number_of_nodes>>
#SBATCH --ntasks-per-node=<<number_of_process_per_node>>
#SBATCH --ntasks=<<total_tasks>>
#SBATCH --cpus-per-task=<<threads_per_process>>
#SBATCH --partition=<<partition>>
#SBATCH --job-name=<<job_name>>
#SBATCH --time=<<max_time>>
#SBATCH --exclusive
#SBATCH --mem=<<max_memory>>

module load nas/1.0

cd $SLURM_SUBMIT_DIR

ulimit -s unlimited
ulimit -c unlimited
ulimit -v unlimited

EXEC="ntb.sh"

PARAMS="<<application_params>>"

if [ $SLURM_NTASKS_PER_NODE -eq 1 ]; then
  CPU_BIND="none"
else 
  CPU_BIND="socket"
fi       

srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK --cpu-bind=$CPU_BIND $EXEC $PARAMS

