# !/bin/sh
COLOR_REST='\e[0m'
COLOR_GREEN='\e[0;32m'
COLOR_BLUE='\e[0;34m'
COLOR_RED='\e[0;31m'

# runner='srun -p prof -N1 -n1 --gres=gpu:1'
runner=''
program="hw4-1_t"
out='test.out'
res="./measure/res.csv"

echo -e "${COLOR_BLUE}Setting up Environment...${COLOR_REST}"
rm ./execs/${program}
rm ${res}

echo -e "${COLOR_BLUE}Compiling...${COLOR_REST}"
cd timer
make
mv ${program} ../execs/${program}
cd ..

# Testcases
tcs=('c10.1' 'c15.1' 'c17.1' 'c18.1' 'c20.1' 'p11k1')
tc_num=${#tcs[@]}

echo "verteice, computing, H2D, D2H, I/O Read, I/O Write, execution time" >> ${res}
for ((idx=0;idx<${tc_num};idx++))
do
    echo -e "${COLOR_GREEN}${idx}-Testcase ${tcs[idx]}${COLOR_REST}"

    # num=0
    # if [ "${tcs[idx]}" -gt 9 ]; then
    #     num=${tcs[idx]}
    # else
    #     num="0${tcs[idx]}"
    # fi
    # rm out/c${num}.1.out

    echo -e "${runner} ./execs/${program} ./sample/cases/${tcs[idx]} ./out/${out} >> ${res}"
    ${runner} ./execs/${program} ./sample/cases/${tcs[idx]} ./out/${out} >> ${res}
done

rm ./out/${out}
echo -e "${COLOR_GREEN}DONE${COLOR_REST}"