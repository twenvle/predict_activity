#!/bin/bash
#$ -cwd
#$ -l cpu_16=1
#$ -l h_rt=23:00:00
#$ -V

. /etc/profile.d/modules.sh
module load gaussian

g16 pa_ts.gjf && formchk pa_ts.chk pa_ts.fchk