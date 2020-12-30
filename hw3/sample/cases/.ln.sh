#!/bin/sh
DIR=/home/pp20/share/.testcase/hw3


for dir in $DIR/*; do
	echo $dir
	echo $(basename "${dir}")
	ln -s $dir $(basename "${dir}")
done
