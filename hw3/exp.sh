# !/bin/bash
dir="./other_ver/timer"
report_dir="./measure"
execs="./execs"
hw3="hw3-t"
analy_post="-a"
load_post="-b"
report="report.csv"
exp_file="exp.csv"
out="_tc"

# Select target
target=${hw3}
in_dir="/home/pp20/share/hw3/cases/"
out_dir="./out/"

# Set up Environment
echo -e "-->Setting up Environment"
mkdir ${execs}
mkdir ${out_dir}
mkdir ${report_dir}
rm ${report_dir}/${exp_file} 
rm ${report_dir}/${exp_file_b}
rm ${report_dir}/${report} 
rm ${report_dir}/${report_b}
echo -e "----------------------------------------"
echo -e ""

# Experiment arguments
runner="srun"
is_show_timer="N"
timer_specs="threads, thread_1_load, thread_2_load, \
             thread_3_load, thread_4_load, thread_5_load, \
             thread_6_load, thread_7_load, thread_8_load, \
             thread_9_load, thread_10_load, thread_11_load, \
             thread_12_load, cpu, memory, I/O, total"
# Testcase 
tcs=(18 18 18 18 18 18)
exp1_tc=18
exp_num=${#tcs[@]}
exp_threads=(1 2 4 6 8 12)

# Compile
echo -e "-->Compiling"
cd ${dir}
make
mv ${hw3} ../.${execs}
cd ../../
echo -e "----------------------------------------"
echo -e ""

# Run Experiments
echo -e "-->Conducting Experiment1"
echo -e ""

echo -e ${timer_specs} >> ${report_dir}/${exp_file}
for ((idx=0;idx<${exp_num};idx++))
do
    # Get name of the testcase
    num=0
    if [ "${tcs[idx]}" -gt 9 ]; then
        num=${tcs[idx]}
    else
        num="0${tcs[idx]}"
    fi
    rm out/c${num}.1.out

    echo -e "--->Experiment ${idx}: ${exp_threads[idx]} Threads in 1 Process"
    echo -e "${runner} -n1 -c${exp_threads[idx]} ${execs}/${target} ${in_dir}/c${num}.1 ${out_dir}c${num}.1.out"
    echo -e "Results"
    echo -e ""

    # Execute program
    # exp_res=`${execs}/${target} ${out_dir}/${target}${out}${idx}.png ${tcs[0]} ${report_dir}/${report}`
    # ${execs}/${target} ${out_dir}/${target}${out}${idx}.png ${tcs[idx]} ${report_dir}/${report}
    if [ "$is_show_timer" = "Y" ]; then
        ${runner} -n1 -c${exp_threads[idx]} ${execs}/${target} ${in_dir}/c${num}.1 ${out_dir}c${num}.1.out
    else
        # exp_res=`${runner} -n1 -c${exp_threads[idx]} ${execs}/${target} ${in_dir}/c${num}.1 ${out_dir}c${num}.1.out ${report_dir}/${report} ${is_show_timer}`
        exp_res=`${runner} -n1 -c${exp_threads[idx]} ${execs}/${target} ${in_dir}/c${num}.1 ${out_dir}c${num}.1.out ${report_dir}/${exp_file} ${is_show_timer}`
        echo -e "${timer_specs}"
        echo -e "${exp_res}"

        commas=""
        twl=12
        for ((com_i=0;com_i<`expr $twl - ${exp_threads[$idx]}`;com_i++))
        do 
            commas="${commas}0.0, "
        done
        echo -e "${exp_threads[idx]}, ${commas} ${exp_res}" >> ${report_dir}/${exp_file}
    fi
    echo -e ""

    echo -e "----------------------------------------"
    echo -e ""
done


# # Experiment2
# target=${hw2b}
# thread=6
# runner="srun"
# exp_procs=(2 3 4 6 8)
# exp_num=5
# timer_specs="processes, process_1_load, process_2_load, \
#              process_3_load, process_4_load, process_5_load, \
#              process_6_load, process_7_load, \
#              cpu, communication, I/O, total"
# # timer_specs="processes, cpu, communication, I/O, total"
# tcs=('174170376 -0.7894722222222222 -0.7825277777777778 0.145046875 0.148953125 2549 1439')
# # tcs=('1000 -1 1 -1 1 1600 1600')

# echo -e "-->Conducting Experiment2"
# report=${report_b}
# exp_file=${exp_file_b}

# echo -e ${timer_specs} >> ${report_dir}/${exp_file}
# for ((idx=0;idx<${exp_num};idx++))
# do
#     echo -e "--->Experiment ${idx}: ${exp_procs[idx]} Processes with ${thread} Threads per process"
#     echo -e "${runner} -n${exp_procs[idx]} -c${thread} ${execs}/${target} ${out_dir}/${target}${out}${idx}.png ${tcs[0]} ${report_dir}/${report} ${is_show_timer}"
#     echo -e "Results"
#     echo -e ""
#     # exp_res=`${execs}/${target} ${out_dir}/${target}${out}${idx}.png ${tcs[0]} ${report_dir}/${report}`
#     # ${execs}/${target} ${out_dir}/${target}${out}${idx}.png ${tcs[idx]} ${report_dir}/${report}
#     if [ "$is_show_timer" = "Y" ]; then
#         ${runner} -n${exp_procs[idx]} -c${thread} ${execs}/${target} ${out_dir}/${target}${out}${idx}.png ${tcs[0]} ${report_dir}/${report} ${is_show_timer}
#     else
#         exp_res=`${runner} -n${exp_procs[idx]} -c${thread} ${execs}/${target} ${out_dir}/${target}${out}${idx}.png ${tcs[0]} ${report_dir}/${report} ${is_show_timer}`
#         echo -e "${timer_specs}"
#         echo -e "${exp_res}"

#         commas=""
#         twl=8
#         for ((com_i=0;com_i<`expr $twl - ${exp_procs[$idx]}`;com_i++))
#         do 
#             commas="${commas}0.0, "
#         done
#         echo -e "${exp_procs[idx]}, ${commas} ${exp_res}" >> ${report_dir}/${exp_file}
#     fi
#     echo -e ""

#     echo -e "----------------------------------------"
#     echo -e ""
# done