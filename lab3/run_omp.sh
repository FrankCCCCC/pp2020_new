# !/bin/bash

g++ -lm ./sample/lab3_omp.cc -o ./sample/lab3_omp -fopenmp
g++ -lm ./lab3_omp.cc -o ./lab3_omp -fopenmp

tc_n=2
r=(10000, 1067212, 9183439)
k=(199, 555, 823)

# tc_n = $(( tc_n - 1))

for i in `seq 0 ${tc_n}`
do
    echo -e "Testcase ${i}: r=${r[i]} k=${k[i]}"

    res_std=`srun -c4 -n1 ./sample/lab3_omp ${r[i]} ${k[i]}`
    res_self=`srun -c4 -n1 ./lab3_omp ${r[i]} ${k[i]}`
    # res_self=res_self+1

    if [ $res_std -eq $res_self ]
    then
        echo -e "Testcase ${i} Passed, Result: ${res_self}"
    else
        echo -e "Testcase ${i} Failed, Result: ${res_self}, Correct: ${res_std}"
    fi
done