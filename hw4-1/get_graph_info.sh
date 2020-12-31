# !/bin/bash

g++ -O3 get_graph_info.c -o execs/get_graph_info
path="/home/pp20/share/hw4-2/cases/c${1}.1"
echo -e "Graph c${1}.1 from ${path}"

execs/get_graph_info ${path}