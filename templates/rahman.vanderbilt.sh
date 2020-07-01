{% extends "base_script.sh" %}
{% block header %}
#!/bin/sh -l
#PBS -j oe
#PBS -l nodes=1:ppn=16
#PBS -l walltime=96:00:00
#PBS -q standard
#PBS -V
#PBS -m abe
#PBS -o /raid6/homes/firstcenter/chloroform/output/

source activate new-signac
module load gromacs/2018.5

{% endblock %}
