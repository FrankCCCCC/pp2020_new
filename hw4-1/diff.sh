# !/bin/bash
echo "${0} Testcase ${1}, Size ${2}"

echo "Compiling"
g++ diff.c -o ./execs/diff

num=0
if [ "${1}" -gt 9 ]; then
    num=${tcs[idx]}
else
    num="0${1}"
fi

echo -e "./execs/diff ${2} out/c${num}.1.out sample/cases/c${num}.1.out"
echo -e "---\n"

./execs/diff ${2} out/c${num}.1.out sample/cases/c${num}.1.out