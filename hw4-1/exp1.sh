# !/bin/sh

function join_by { local d=$1; shift; local f=$1; shift; printf %s "$f" "${@/#/$d}"; }

COLOR_REST='\e[0m'
COLOR_GREEN='\e[0;32m'
COLOR_BLUE='\e[0;34m'
COLOR_RED='\e[0;31m'

# runner='srun -p prof -N1 -n1 --gres=gpu:1'
runner=''
programs=("hw4-1_t" "hw4-1_nostream_t" "hw4-1_global_t")
# progs=join_by ' ' "${programs[@]}"
out='test.out'
res="./measure/res1.csv"

echo -e "Targets ${programs}"

echo -e "${COLOR_BLUE}Setting up Environment...${COLOR_REST}"
for ((idx=0;idx<${#programs[@]};idx++))
do
    rm ./execs/${programs[idx]}
done
rm ${res}

echo -e "${COLOR_BLUE}Compiling...${COLOR_REST}"
cd timer
make
for ((idx=0;idx<${#programs[@]};idx++))
do
    mv ${programs[idx]} ../execs/
done
cd ..

# Testcases
# tcs=('c10.1' 'c15.1' 'c17.1' 'c18.1' 'c20.1' 'p11k1')
# tc_num=${#tcs[@]}
tc=('c20.1')
prog_num=${#programs[@]}

echo "verteice, computing, H2D, D2H, I/O Read, I/O Write, execution time" >> ${res}
for ((idx=0;idx<${prog_num};idx++))
do
    echo -e "${COLOR_GREEN}${idx}-Program ${programs[idx]}${COLOR_REST}"
    program=${programs[idx]}
    # num=0
    # if [ "${tcs[idx]}" -gt 9 ]; then
    #     num=${tcs[idx]}
    # else
    #     num="0${tcs[idx]}"
    # fi
    # rm out/c${num}.1.out

    echo -e "${runner} ./execs/${program} ./sample/cases/${tc} ./out/${out} >> ${res}"
    ${runner} ./execs/${program} ./sample/cases/${tc} ./out/${out} >> ${res}
done

rm ./out/${out}
echo -e "${COLOR_GREEN}DONE${COLOR_REST}"