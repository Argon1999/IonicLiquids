{% extends "base_script.sh" %}
{% block header %}
#!/bin/bash
#SBATCH --job-name=em
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time={{ walltime|format_timedelta }}
#SBATCH -C haswell
#SBATCH --error=error.err
#SBATCH --output=/global/homes/b/bansaa2/IonicLiquids/output/
#SBATCH -q regular


module load gromacs/2020.1.hsw

{% endblock %}

