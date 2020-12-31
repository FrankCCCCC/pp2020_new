# !bin/bash
bash compile.sh
path="/home/pp20/share/hw4-2/cases/c${1}.1"
srun -N1 -n1 -c2 --gres=gpu:2 execs/hw4-2 hades:${path} c${1}.1