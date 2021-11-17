
for i in 10000 100000 1000000 10000000 100000000
do
  for j in 16 32 64 128 256
    do
        ./ex_3 $i $j
    done
done
