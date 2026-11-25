#!/bin/bash
#SBATCH --job-name=chroma_baseg_cpu
#SBATCH --partition=lq3_cpu
#SBATCH --account=nplqcd.lq3_cpu
#SBATCH --qos=normal
#SBATCH --nodes=6
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=/lustre2/nplqcd/vayyar/grid_qcd/logs/chroma_baseg_cpu_%j.log
#
# Chroma hmc on 6 lq3 nodes (the production allocation for Grid, user 11-23): 8 ranks per node,
# 8 threads each (one rank per 8-core CCD, as Grid's production layout), 48 ranks, QMP geometry
# 2.2.2.6: local volume 24.24.24.16 (X a multiple of 8 for QPhiX's soalen 4, Y and Z multiples of
# QPhiX's block sizes 4). Settings and steps: chroma_baseg_cpu.sh. Every submission is approved by
# the user, e.g. trajectory 2001 from cfg_2000 (seeds 300 ...):
#   N_TRAJ=1 sbatch --time=24:00:00 --export=ALL .../chroma_inputs/lq/chroma_baseg_cpu_job.sh
exec bash /lustre2/nplqcd/vayyar/grid_qcd/grid-lqcd-workflow/5_studies/chroma_inputs/lq/chroma_baseg_cpu.sh
