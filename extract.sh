#!/bin/bash

# If args is "seq-prop", then run sequential proportion experiment
if [ "$1" == "seq-prop" ]; then
    # Print floating points from 0.1 to 0.9 with 0.1 increment
    for i in $(seq 0.05 0.05 0.95); do
        python oselm/train-os-elm.py sample fashion-mnist cuda $i
    done

    for i in $(seq 0.96 0.01 0.99); do
        python oselm/train-os-elm.py sample fashion-mnist cuda $i
    done
    exit 0
fi

if [ "$1" == "batch-size" ]; then
    python oselm/train-os-elm.py sample fashion-mnist
    for i in $(seq 2 1 4); do
        python oselm/train-os-elm.py batch "$i" fashion-mnist
    done

    for i in $(seq 5 5 100); do
        python oselm/train-os-elm.py batch "$i" fashion-mnist
    done
    exit 0
fi
