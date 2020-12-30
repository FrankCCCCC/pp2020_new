# !bin/bash
bash compile.sh
srun -N1 -n1 -c2 --gres=gpu:2 execs/hw4-2 hades:/home/pp20/share/hw4-2/cases/c21.1 c21.1