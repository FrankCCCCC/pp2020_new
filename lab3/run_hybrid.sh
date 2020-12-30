# !/bin/bash


g++ ./sample/lab3_pthread.cc -o ./sample/lab3_pthread -pthread -lm
mpicxx lab3_hybrid.cc -o lab3_hybrid -fopenmp -lm

tc_n=5
r=(10000, 1067212, 9183439, 1781232, 212125892, 1401149118)
k=(199, 555, 823, 101, 101, 14011491183)

# tc_n=5
# r=(10, 106, 9183, 1782, 29138, 1401)
# k=(1000000000, 10000000000000, 100000000000000, 100000000000000, 100000000000000, 100000000000000)
# k=(1000000210, 100000000000011, 100000000002001, 100001030000001, 100000000000201, 100000000000011)

# tc_n=0
# r=(10)
# k=(101)

# tc_n = $(( tc_n - 1))

N=1
c=8
n=1

echo -e "Parameters: -N${N} -c${c} -n${n}"

for i in `seq 0 ${tc_n}`
do
    echo -e "Testcase ${i}: r=${r[i]} k=${k[i]}"

    res_std=`srun -N${N} -c${c} -n${n} ./sample/lab3_pthread ${r[i]} ${k[i]}`
    res_self=`srun -N${N} -c${c} -n${n} ./lab3_hybrid ${r[i]} ${k[i]}`
    # res_self_time=`time srun -c1 -n1 ./lab3_hybrid ${r[i]} ${k[i]}`
    # res_self=res_self+1

    if [ $res_std -eq $res_self ]
    then
        echo -e "Testcase ${i} Passed, Result: ${res_self}, Correct: ${res_std}\n"
    else
        echo -e "Testcase ${i} Failed, Result: ${res_self}, Correct: ${res_std}\n"
    fi
done

# srun -N1 -c4 -n1 ./lab3_hybrid 9183439 823