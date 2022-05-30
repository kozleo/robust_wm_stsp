#!/bin/bash

for k in 5 10 15; do
    for b in 20 50 100; do
        python decode_sample_mp.py $k $b;
    done
done