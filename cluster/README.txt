runjob.sh is a sample submission script for use on the Sphynx cluster.

In order to copy files to and from the Sphynx cluster using the username
fhoffma, use:

scp -r ./Documents/python/code/ fhoffma@sphynx.ccs.tulane.edu:/scratch03/fhoffma

scp -r fhoffma@sphynx.ccs.tulane.edu:/scratch03/fhoffma/python/code/results_tmp ./Documents/python/code/results_tmp/
