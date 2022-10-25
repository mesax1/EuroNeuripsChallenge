#!/bin/bash
#SBATCH --job-name neurips
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --partition=longjobs
#SBATCH -o test_%N_%j.out
#SBATCH -e test_%N_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=auribev1@eafit.edu.co
#SBATCH --array=0-249
#SBATCH --nodes=1
##SBATCH --exclusive=user
##SBATCH --mem=8g
export OMP_NUM_THREADS=$SLURM_NTASKS

module load python-3.9.9-gcc-11.2.0-k7juzmi

### Only run once to create the virtualenv correctly###
#virtualenv -p python3.9 env
#source env/bin/activate
#./install.sh

### Activate ENV ###
source env/bin/activate

##### JOB COMMANDS ####
#python results.py w
python controller.py --instance 0 --epoch_tlim 60 -- python solver.py --verbose --strategy knearestlast,1
python controller.py --instance 0 --epoch_tlim 60 -- python solver.py --verbose --strategy knearestlast,1.2
python controller.py --instance 0 --epoch_tlim 60 -- python solver.py --verbose --strategy knearestlast,1.4
python controller.py --instance 0 --epoch_tlim 60 -- python solver.py --verbose --strategy knearestlast,1.6
python controller.py --instance 0 --epoch_tlim 60 -- python solver.py --verbose --strategy knearestlast,1.8
python controller.py --instance 0 --epoch_tlim 60 -- python solver.py --verbose --strategy knearestlast,2
python controller.py --instance 0 --epoch_tlim 60 -- python solver.py --verbose --strategy knearestlast,2.2
python controller.py --instance 0 --epoch_tlim 60 -- python solver.py --verbose --strategy knearestlast,2.4
python controller.py --instance 0 --epoch_tlim 60 -- python solver.py --verbose --strategy knearestlast,2.6
python controller.py --instance 0 --epoch_tlim 60 -- python solver.py --verbose --strategy knearestlast,2.8
python controller.py --instance 0 --epoch_tlim 60 -- python solver.py --verbose --strategy knearestlast,3
#python controller.py --instance $SLURM_ARRAY_TASK_ID --epoch_tlim 240 --static -- python solver.py --verbose --strategy modifiedknearest
#python results.py r modifiedknearest