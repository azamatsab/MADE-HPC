#!/bin/bash

END=1000
for ((n=100;n<=END;n++));
do
   out=$(( $n % 2 ))
   if [ $out -eq 0 ] 
   then
	echo $n
   fi	
done