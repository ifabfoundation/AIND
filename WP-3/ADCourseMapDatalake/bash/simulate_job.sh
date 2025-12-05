#!/bin/bash
#SBATCH --job-name=ADCourseMap
#SBATCH --output=/home/IFAB/WORK/ADCourseMap/results/ADCourseMap_simulate_%j.out
#SBATCH --error=/home/IFAB/WORK/ADCourseMap/results/ADCourseMap_simulate_%j.err
#SBATCH --nodes=1                  # Numero di nodi da utilizzare
#SBATCH --ntasks-per-node=1        # Processi per nodo
#SBATCH --cpus-per-task=3          # CPU per processo (per calcoli più intensivi)
#SBATCH --time=03:00:00            # Tempo massimo di esecuzione (3h)
#SBATCH --mem=24G                   # Memoria per nodo

# Attiva l'ambiente Python
source /home/IFAB/WORK/ADCourseMap/myvenv/bin/activate

# Mostra informazioni sul job
echo "Job SLURM: $SLURM_JOB_ID avviato su $SLURM_JOB_NODELIST"
echo "Totale processi: $SLURM_NTASKS"
echo "Utente: $USER"
echo "Data e ora di inizio: $(date)"

# Esegui lo script Python sui nodi assegnati
# Ogni processo riceverà un SLURM_PROCID univoco
#srun python /home/IFAB/WORK/ADCourseMap/code/main.py --n_iter 100000 --device cpu --level cleaned_02 --n_burn_in_iter_frac 0.7 --burn_in_step_power 0.9 --source_dimension 3 
srun python /home/IFAB/WORK/ADCourseMap/code/main.py --level cleaned_03 --device cpu --file_code ADNIMERGE --same_data_stats 1 --simulation True

echo "Job completato: $(date)"
