# !/bin/bash
echo -e "$0 $1"

# Sequential Version
seq_dir="./sample"
seq="hw2seq"

# Executable Files
execs="./execs"
# target_dir="./execs"
target=""
runner=""

if [ "$1" = "hw2a" ]; then
    target="hw2a"
    runner="srun"
    runner_params='-n1 -c6'
elif [ "$1" = "hw2b" ]; then 
    target="hw2b"
    runner="srun"
    runner_params='-n8 -c6'
else
    echo -e "Please specify the tareget file 'hw2a' or 'hw2b'"
    exit 1
fi

# Output Files
out_dir="./out/${target}"
out="_tc"
tc_num=6
mkdir ${out_dir}

# Remove Old Files
echo -e "----------------------------------------"
echo -e "Removing Old Files"
rm ${execs}/${target} ${execs}/${seq}
for ((idx=0;idx<${tc_num};idx++))
do
    rm ${out_dir}/${target}${out}${idx}.png ${out_dir}/${seq}${out}${idx}.png
    # echo "${idx}"
done

# Compile
echo -e "----------------------------------------"
echo -e "Compiling"
bash ./compile.sh
cd ${seq_dir}
make
mv ${seq} .${execs}
cd ../

# Run Testcase 
params=('2602 -3 0.2 -3 0.2 979 2355' \
        '3000 -1 1 -1 1 1600 1600' \
        '18667 -2 2 -2 2 575 575'  \
        '10000 -5 6 -7 8 56 78' \
        '5000 -5 6 -3 4 1600 1600'\
        '174170376 -0.7894722222222222 -0.7825277777777778 0.145046875 0.148953125 2549 1439'
        )

echo -e "----------------------------------------"
echo -e ""
for ((idx=0;idx<${tc_num};idx++))
do
    echo -e "Testcase${idx} ${params[idx]}"
    echo -e ""

    echo -e "-->Running Sequential Version"
    echo -e "${execs}/${seq} ${out_dir}/${seq}${out}${idx}.png ${params[idx]}"
    ${execs}/${seq} ${out_dir}/${seq}${out}${idx}.png ${params[idx]}
    echo -e ""

    echo -e "-->Running Pthread Version"
    echo -e "${runner} ${runner_params} ${execs}/${target} ${out_dir}/${target}${out}${idx}.png ${params[idx]}"
    ${runner} ${runner_params} ${execs}/${target} ${out_dir}/${target}${out}${idx}.png ${params[idx]}
    echo -e ""

    echo -e "-->Diff the Output"
    echo -e "hw2-diff ${out_dir}/${seq}${out}${idx}.png ${out_dir}/${target}${out}${idx}.png"
    hw2-diff ${out_dir}/${seq}${out}${idx}.png ${out_dir}/${target}${out}${idx}.png
    echo -e ""

    echo -e "----------------------------------------"
    echo -e ""
    # echo -e ""
done