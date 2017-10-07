# A Numerical Method for Doubly-Periodic Stokes Flow in 3D with and without a Bounding Plane

In my PhD thesis ([add link]) I derived a fast numerical method to compute Stokes flow in 3D that is periodic in two directions.
Here is my implementation.

* If you want to recreate the data and figures from my thesis, have a look at the [figures/thesis](figures/thesis) folder.

* If you want to play with the method I derived, have a look at [example.py](example.py) and the comments therein. 
To run, simply type 
```bash
python2.7 example.py
```
With the current settings this will create the following flow field:
[add pic]


### Running on the Sphynx cluster

Some computations were quite intensive. I ran those on the [Sphynx cluster](https://www2.tulane.edu/sse/ccs/computing/hardware.cfm) of Tulane University's Center for Computational Science. Here's how:

1. Copy the files you want to run from your local machine to Sphynx, e.g. 
```bash
scp -r ./Documents/python/ fhoffma@sphynx.ccs.tulane.edu:/scratch03/fhoffma/python
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

