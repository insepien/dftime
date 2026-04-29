#!/bin/bash -l

#SBATCH --account=kmpardo_1142
#SBATCH --mail-type=END
#SBATCH --mail-user=hamydo@usc.edu

#SBATCH --partition=main
#SBATCH --job-name=figaro_run       # Job name
#SBATCH --output=log_%j.out         # Standard output and error log (%j = jobID)

#SBATCH --nodes=1                   # Run on a single node
#SBATCH --ntasks=1                  # Single task
#SBATCH --cpus-per-task=16          # Number of CPU cores (Adjust based on cluster)
#SBATCH --mem=24G                   # Total memory
#SBATCH --time=6:00:00             # Time limit (hrs:min:sec)



# 1. Load your environment
module purge
eval "$(conda shell.bash hook)"
conda activate agn
cd /home1/hamydo/dftime

# 2. Set the Python Path so it finds your 'src' folder
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 3. Run FIGARO with parallelization
# --n_parallel: number of threads for hierarchical inference
# --n_parallel_events: number of threads for single-event draws

figaro-par-hierarchical \
    -i /scratch1/hamydo/dftime_data/events \
    -b "[[0,40],[0,100]]" \
    --se_draws 100 \
    --draws 200 \
    --selfunc script/selfunc_aq.py \
    --n_parallel 16