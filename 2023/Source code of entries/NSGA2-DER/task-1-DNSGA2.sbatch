#!/bin/bash
#SBATCH -J task-1-DNSGA2.sbatch
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH -o ./slrum/%j.out

python main.py 1 DNSGA2