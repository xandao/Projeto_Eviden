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

# Other params: -m GTRGAMMA -p 112233  -b 223344 -c 4 -f d

PARAMS="-T $SLURM_CPUS_PER_TASK -N <<bootstrap>> -n <<output_file>> -s <<input_file>> <<other_params>>

if [ $SLURM_NTASKS_PER_NODE -eq 1 ]; then
  CPU_BIND="none"
else 
  CPU_BIND="socket"
fi       

srun -n $SLURM_NTASKS -c $SLURM_CPUS_PER_TASK --cpu-bind=$CPU_BIND $EXEC $PARAMS

