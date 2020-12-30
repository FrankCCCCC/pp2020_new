# !/bin/bash
echo -e "Compiling..."
g++ gen.cc -o ./execs/gen
echo -e "Generating Testcase..."
./execs/gen
echo -e "Move Testcase to ./gen_testcase..."
mv ./mycase.in ./gen_testcase/mycase.in

# Validate
echo -e "Validating..."
g++ ./sample/validator.cc -o ./execs/validator
./execs/validator ./gen_testcase/mycase.in

# Run
echo -e "Solving..."
bash compile.sh
time ./execs/hw3 ./gen_testcase/mycase.in ./out/mycase.out