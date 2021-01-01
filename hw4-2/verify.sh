# !/bin/bash
program="hw4-2"

echo -e "Setting up Environment..."
mkdir out
rm ./execs/${program}
rm ./out/c*

echo -e "Compiling..."
bash compile.sh

# tc_dir='/home/pp20/share/hw4-1/cases/'
tc_dir='./sample/s_cases/'
# runner='srun -n 1 --gres=gpu:1'
runner=''
# tcs=(0 1)
tcs=(0 1 2 3 4 5 6 7 15 17)
# tcs=(0 1 2 3 4 5 6 7 15)
# tcs=(0 17 18)
# tcs=(0 21)
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