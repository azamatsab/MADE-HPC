#!/bin/bash

array=(1 2 4 8 16 32 64 128 512 1024 2048 4096 8192)

for t in ${array[@]}; do
  echo $t
done
