#!/bin/bash
#SBATCH --job-name=01_Bridge
#SBATCH --ntasks=16
#SBATCH --time=01:00:00
#SBATCH --output=pw.out


srun pw.x -in pw.in > pw.out
