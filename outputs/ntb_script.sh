#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=14
#SBATCH --partition=sequana_cpu
#SBATCH --job-name=teste2
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --mem=350G

module load nas/1.0

cd $SLURM_SUBMIT_DIR

ulimit -s unlimited
ulimit -c unlimited
ulimit -v unlimited

EXEC="ntb.sh"

PARAMS="-b bt-mz -c A"

if [ $SLURM_NTASKS_PER_NODE -eq 1 ]; then
  CPU_BIND="none"
else 
  CPU_BIND="socket"
fi       

srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK --cpu-bind=$CPU_BIND $EXEC $PARAMS

