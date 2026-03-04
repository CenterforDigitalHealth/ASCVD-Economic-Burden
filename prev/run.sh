#!/bin/bash

python HMM_main.py -t 1 -m 1 -i 0.11 -d 0.02 -s 'val' 2>&1 | tee logs/log_t1m1i1d2v.txt &
python HMM_main.py -t 1 -m 1 -i 0.11 -d 0.02 -s 'upper' 2>&1 | tee logs/log_t1m1i1d2u.txt &
python HMM_main.py -t 1 -m 1 -i 0.11 -d 0.02 -s 'lower' 2>&1 | tee logs/log_t1m1i1d2l.txt &

python HMM_main.py -t 1 -m 1 -i 0.11 -d 0 -s 'val' 2>&1 | tee logs/log_t1m1i1d0v.txt &
python HMM_main.py -t 1 -m 1 -i 0.11 -d 0 -s 'upper' 2>&1 | tee logs/log_t1m1i1d0u.txt &
python HMM_main.py -t 1 -m 1 -i 0.11 -d 0 -s 'lower' 2>&1 | tee logs/log_t1m1i1d0l.txt &

python HMM_main.py -t 1 -m 1 -i 0.11 -d 0.03 -s 'val' 2>&1 | tee logs/log_t1m1i1d3v.txt &
python HMM_main.py -t 1 -m 1 -i 0.11 -d 0.03 -s 'upper' 2>&1 | tee logs/log_t1m1i1d3u.txt &
python HMM_main.py -t 1 -m 1 -i 0.11 -d 0.03 -s 'lower' 2>&1 | tee logs/log_t1m1i1d3l.txt &

python HMM_main.py -t 1 -m 1 -i 0.05 -d 0.02 -s 'val' 2>&1 | tee logs/log_t1m1i5d2v.txt &
python HMM_main.py -t 1 -m 1 -i 0.05 -d 0.02 -s 'upper' 2>&1 | tee logs/log_t1m1i5d2u.txt &
python HMM_main.py -t 1 -m 1 -i 0.05 -d 0.02 -s 'lower' 2>&1 | tee logs/log_t1m1i5d2l.txt &

python HMM_main.py -t 1 -m 1 -i 0.23 -d 0.02 -s 'val' 2>&1 | tee logs/log_t1m1i2d2v.txt &
python HMM_main.py -t 1 -m 1 -i 0.23 -d 0.02 -s 'upper' 2>&1 | tee logs/log_t1m1i2d2u.txt &
python HMM_main.py -t 1 -m 1 -i 0.23 -d 0.02 -s 'lower' 2>&1 | tee logs/log_t1m1i2d2l.txt &

python HMM_main.py -t 1 -m 1 -i 0 -d 0.02 -s 'val' 2>&1 | tee logs/log_t1m1i0d2v.txt &
python HMM_main.py -t 1 -m 1 -i 0 -d 0.02 -s 'upper' 2>&1 | tee logs/log_t1m1i0d2u.txt &
python HMM_main.py -t 1 -m 1 -i 0 -d 0.02 -s 'lower' 2>&1 | tee logs/log_t1m1i0d2l.txt &

python HMM_main.py -t 0 -m 1 -i 0 -d 0.02 -s 'val' 2>&1 | tee logs/log_t0m1i0d2v.txt &
python HMM_main.py -t 0 -m 1 -i 0 -d 0.02 -s 'upper' 2>&1 | tee logs/log_t0m1i0d2u.txt &
python HMM_main.py -t 0 -m 1 -i 0 -d 0.02 -s 'lower' 2>&1 | tee logs/log_t0m1i0d2l.txt &

python HMM_main.py -t 0 -m 1 -i 0.11 -d 0.02 -s 'val' 2>&1 | tee logs/log_t0m1i1d2v.txt &
python HMM_main.py -t 0 -m 1 -i 0.11 -d 0.02 -s 'upper' 2>&1 | tee logs/log_t0m1i1d2u.txt &
python HMM_main.py -t 0 -m 1 -i 0.11 -d 0.02 -s 'lower' 2>&1 | tee logs/log_t0m1i1d2l.txt &

python HMM_main.py -t 1 -m 1 -i 0 -d 0 -s 'val' 2>&1 | tee logs/log_t1m1i0d0v.txt &
python HMM_main.py -t 1 -m 1 -i 0 -d 0 -s 'upper' 2>&1 | tee logs/log_t1m1i0d0u.txt &
python HMM_main.py -t 1 -m 1 -i 0 -d 0 -s 'lower' 2>&1 | tee logs/log_t1m1i0d0l.txt &

python HMM_main.py -t 0 -m 1 -i 0 -d 0 -s 'val' 2>&1 | tee logs/log_t0m1i0d0v.txt &
python HMM_main.py -t 0 -m 1 -i 0 -d 0 -s 'upper' 2>&1 | tee logs/log_t0m1i0d0u.txt &
python HMM_main.py -t 0 -m 1 -i 0 -d 0 -s 'lower' 2>&1 | tee logs/log_t0m1i0d0l.txt &

python HMM_main.py -t 0 -m 1 -i 0.0075 -d 0.02 -s 'val' 2>&1 | tee logs/log_t0m1i7d2v.txt &
python HMM_main.py -t 0 -m 1 -i 0.0075 -d 0.02 -s 'upper' 2>&1 | tee logs/log_t0m1i7d2u.txt &
python HMM_main.py -t 0 -m 1 -i 0.0075 -d 0.02 -s 'lower' 2>&1 | tee logs/log_t0m1i7d2l.txt &

## To view the progress of each process, run the following command to 
## cat logs/log_t0m1l0.txt | grep "\-" | wc -l # target output, 1
## tail logs/log_t0m1l0.txt  # target output, Z**
