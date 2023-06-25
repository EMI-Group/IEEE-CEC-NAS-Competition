#!/bin/bash
#SBATCH -J #NAME#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 6
#SBATCH -o ./slrum/%j.out

python main.py #PID# #METHOD#