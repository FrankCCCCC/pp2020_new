# !/bin/bash
program="hw4-1"

echo -e "Setting up Environment..."
rm ./execs/${program}

echo -e "Compiling..."
bash compile.sh

# runner='srun -n 1 --gres=gpu:1'
runner=''
tc_dir='./sample/s_cases/'
# tcs=(0 4)
tcs=(0 1 2 3 4 5 6 7 15 17)
# tcs=(0 1 2 3 4 5 6 7 15)
# tcs=(0 17 18)
# tcs=(0 1)
tc_num=${#tcs[@]}
# vatc_numr=$((tc_num+1))

for ((idx=1;idx<${tc_num};idx++))
do
    echo -e "${idx}-Testcase ${tcs[idx]}"

    num=0
    if [ "${tcs[idx]}" -gt 9 ]; then
        num=${tcs[idx]}
    else
        num="0${tcs[idx]}"
    fi
    rm out/c${num}.1.out

    echo -e "./execs/${program} ${tc_dir}c${num}.1 ./out/c${num}.1.out"
    
    time ${runner} ./execs/${program} ${tc_dir}c${num}.1 ./out/c${num}.1.out
    echo -e "Diff Result:"
    diff ${tc_dir}c${num}.1.out out/c${num}.1.out
    # diff sample/cases/c0${tcs[idx]}.1 out/c0${tcs[idx]}.1.out
    echo -e "----------------------------------------"
    echo -e ""
done