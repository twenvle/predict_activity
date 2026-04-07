#!/bin/bash
#$ -cwd
#$ -l cpu_16=1
#$ -l h_rt=23:00:00
#$ -V

. /etc/profile.d/modules.sh
module load gaussian

g16 phthalicacid_ts.gjf && formchk phthalicacid_ts.chk phthalicacid_ts.fchk