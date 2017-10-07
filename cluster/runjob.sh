#PBS -q ccs_short
#PBS -N N1L1
#PBS -l walltime=00:30:00
#PBS -l nodes=1:ppn=1
#PBS -d /scratch03/fhoffma/python/code

echo Start Job

module load intel
#module load python/2.7
module load Anaconda

python < cilia_velocity.py 

pwd

echo End Job

