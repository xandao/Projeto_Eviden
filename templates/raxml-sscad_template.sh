#SBATCH --nodes=<<muber_of_nodes>>
#SBATCH --ntasks-per-node=<<munber_of_process_per_node>>
#SBATCH --ntasks=<<ntasks>>
#SBATCH --cpus-per-task=<<threads_per_cpu>>
#SBATCH --partition=<<partition>>
#SBATCH --job-name=<<job_name>>
#SBATCH --time=<<max_time>>
#SBATCH --exclusive
#SBATCH --mem=<<max_memory>>


module load raxml

cd $SLURM_SUBMIT_DIR

ulimit -s unlimited
ulimit -c unlimited
ulimit -v unlimited

EXEC="/petrobr/app_sequana/raxml/8.2.12/bin/raxmlHPC-HYBRID-AVX"

PARAMS="-T $SLURM_CPUS_PER_TASK -N <<bootstrap>> -s <<input_file>> <<other_params>>

srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK --cpu-bind=none $EXEC $PARAMS

