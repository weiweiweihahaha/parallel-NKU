# ordi.sh
#!/bin/sh
# PBS -N arm_ordi

pssh -h $PBS_NODEFILE mkdir -p /home/s2212138 1>&2
scp master:/home/s2212138/arm_ordi /home/s2212138
pscp -h $PBS_NODEFILE master:/home/s2212138/arm_ordi /home/s2212138 1>&2
/home/s2212138/arm_ordi
