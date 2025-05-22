#!/bin/bash

for seed in {0..9}
do
  python make_subset.py --dataset cifar100 --seed $seed --percent 99.8
done
