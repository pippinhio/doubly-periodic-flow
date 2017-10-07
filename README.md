# doubly-periodic-flow


## Running on the Sphynx cluster

Some computations were quite intensive. I ran those on the [Sphynx cluster](https://www2.tulane.edu/sse/ccs/computing/hardware.cfm) of Tulane University's Center for Computational Science. Here's how:

1. Copy the files you want to run from your local machine to Sphynx, e.g. 
```bash
scp -r ./Documents/python/code/ fhoffma@sphynx.ccs.tulane.edu:/scratch03/fhoffma
```

2. Then run a submission script like this one here:
```bash
#PBS -q ccs_short
#PBS -N N1L1
#PBS -l walltime=00:30:00
#PBS -l nodes=1:ppn=1
#PBS -d /scratch03/fhoffma/python/src

echo Start Job

module load intel
module load Anaconda

python < cilia_velocity.py 

pwd

echo End Job
```
