# !/bin/bash

COLOR_REST='\e[0m'
COLOR_GREEN='\e[0;32m'
COLOR_BLUE='\e[0;34m'
COLOR_RED='\e[0;31m'

echo -e "Compiling..."
nvcc -O3 -std=c++11 -Xptxas=-v -arch=sm_61  -lm ./prof/hw4-1_prof.cu -o ./execs/hw4-1_prof

# rm prof/prof.nvvp
# srun -p prof -N1 -n1 --gres=gpu:1 nvprof --metrics gld_throughput -o prof/prof.nvvp ./execs/hw4-1_corr /home/pp20/share/hw4-1/cases/c01.1 ./out/c01.1.out
# srun -p prof -N1 -n1 --gres=gpu:1 nvprof --metrics shared_efficiency gld_throughput -o prof/prof16-16.nvvp ./execs/hw4-1 /home/pp20/share/hw4-1/cases/c21.1 ./out/c21.1.out
# srun -p prof -N1 -n1 --gres=gpu:1 nvprof -f --metrics shared_efficiency ./execs/hw4-1 /home/pp20/share/hw4-1/cases/c21.1 ./out/c21.1.out
# srun -p prof -N1 -n1 --gres=gpu:1 nvprof -f --metrics l2_l1_write_throughput,2_l1_write_throughputs,l2_l1_read_throughput,gld_throughput ./execs/hw4-1 /home/pp20/share/hw4-1/cases/c21.1 ./out/c21.1.out

rm -f prof/gld_out.txt

target="./execs/hw4-1_prof"
prof_out="./prof/prof_f"
prof_out_csv="prof_f.csv"
prof_out_csv_dir="./prof/"
post=".nvvp"
out="prof.out"
tc="c10.1"
Bs=(8 16 32 64)
dim_x=8
dim_y=8
# metric="achieved_occupancy,gld_throughput,shared_load_throughput,shared_store_throughput,gld_throughput,gst_throughput,sm_efficiency"
# metric="shared_efficiency"
# metric="achieved_occupancy"

# metrics=(inst_integer achieved_occupancy shared_load_throughput shared_store_throughput gld_throughput gst_throughput sm_efficiency)
metrics=(inst_integer)

metrics_num=${#metrics[@]}
Bs_num=${#Bs[@]}
((total=${metrics_num}*${Bs_num}))
for ((idxi=0;idxi<${metrics_num};idxi++))
do
    for ((idxj=0;idxj<${Bs_num};idxj++))
    do
        met=${metrics[idxi]}
        bs=${Bs[idxj]}
        prof_file="${prof_out}$_${idxi}-${idxj}${post}"
        ((id=${idxi}*${Bs_num}+${idxj}+1))

        echo -e "${COLOR_GREEN}->${id}/${total}${COLOR_REST}"
        echo -e "${COLOR_BLUE}Metric: ${met}"
        echo -e "Param: ${Bs[idxj]} ${dim_x} ${dim_y}${COLOR_REST}"
        # echo -e "srun -p prof -N1 -n1 --gres=gpu:1 nvprof -f -o ${prof_file} --metrics ${met} ${target} /home/pp20/share/hw4-1/cases/${tc} ./out/${out} ${bs} ${dim_x} ${dim_y}"
        # echo -e "srun -p prof -N1 -n1 --gres=gpu:1 nvprof --csv ${prof_out_csv_dir}${bs}_${met}_${prof_out_csv} --metrics ${met} ${target} /home/pp20/share/hw4-1/cases/${tc} ./out/${out} ${bs} ${dim_x} ${dim_y}"
        # srun -p prof -N1 -n1 --gres=gpu:1 nvprof --csv ${prof_out_csv_dir}${bs}_${met}_${prof_out_csv} --metrics ${met} ${target} /home/pp20/share/hw4-1/cases/${tc} ./out/${out} ${bs} ${dim_x} ${dim_y}
        echo -e "srun -p prof -N1 -n1 --gres=gpu:1 nvprof --metrics ${met} ${target} /home/pp20/share/hw4-1/cases/${tc} ./out/${out} ${bs} ${dim_x} ${dim_y}"
        srun -p prof -N1 -n1 --gres=gpu:1 nvprof --metrics ${met} ${target} /home/pp20/share/hw4-1/cases/${tc} ./out/${out} ${bs} ${dim_x} ${dim_y}
        echo -e "-----------------------\n" >> prof_f.txt
        echo -e "${COLOR_BLUE}-----------------------${COLOR_REST}"
    done
done
