#!/bin/bash
#$ -cwd
#$ -l cpu_40=1
#$ -l h_rt=23:00:00
#$ -V

. /etc/profile.d/modules.sh
module load gaussian

g16 test.gjf && formchk test.chk test.fchk