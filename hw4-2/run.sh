# !bin/bash
bash compile.sh

program='hw4-2'
execs_dir='execs/'
temp_dir='/home/pp20/pp20s02/temp/'

input_path="/home/pp20/share/hw4-2/cases/${1}"
out_path="${temp_dir}${1}"

execute_file="${execs_dir}${program}"
target_file="${temp_dir}${program}"

mkdir ${temp_dir}
cp ${execute_file} ${target_file}

echo -e "srun -N1 -n1 -c2 --gres=gpu:2 ${target_file} hades:${input_path} ${out_path}"
srun -N1 -n1 -c2 --gres=gpu:2 ${target_file} hades:${input_path} ${out_path}
rm ${target_file} ${out_path}