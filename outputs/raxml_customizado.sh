#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=8
#SBATCH --partition=sequana_cpu
#SBATCH --job-name=teste2
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --mem=350G

module load raxml

cd $SLURM_SUBMIT_DIR

ulimit -s unlimited
ulimit -c unlimited
ulimit -v unlimited

EXEC="/petrobr/app_sequana/raxml/8.2.12/bin/raxmlHPC-HYBRID-AVX"

PARAMS="-T $SLURM_CPUS_PER_TASK -N 100 -s /home/xandao/Downloads/DENV_3-colombia-BVBRC_genome_sequence.mafft -x 10000"

if [ $SLURM_NTASKS_PER_NODE -eq 1 ]; then
  CPU_BIND="none"
else 
  CPU_BIND="socket"
fi       

srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK --cpu-bind=$CPU_BIND $EXEC $PARAMS

