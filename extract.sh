#!/bin/bash

DATASET="$2"

valid_datasets=("fashion-mnist" "mnist" "cifar10" "cifar100" "super-tiny-imagenet" "tiny-imagenet")

is_valid_dataset() {
    for d in "${valid_datasets[@]}"; do
        if [ "$d" == "$1" ]; then
            return 0
        fi
    done
    return 1
}

if ! is_valid_dataset "$DATASET"; then
    echo "Invalid dataset: $DATASET"
    exit 1
fi

# If args is "seq-prop", then run sequential proportion experiment
if [ "$1" == "seq-prop" ]; then
    # Print floating points from 0.1 to 0.9 with 0.1 increment
    for i in $(seq 0.05 0.05 0.95); do
        python oselm/train-os-elm.py sample "$DATASET" cuda $i
    done

    for i in $(seq 0.96 0.01 0.99); do
        python oselm/train-os-elm.py sample "$DATASET" cuda $i
    done
    exit 0
fi

if [ "$1" == "batch-size" ]; then
    python oselm/train-os-elm.py sample "$DATASET"
    for i in $(seq 2 1 4); do
        python oselm/train-os-elm.py batch "$i" "$DATASET"
    done

    for i in $(seq 5 5 100); do
        python oselm/train-os-elm.py batch "$i" "$DATASET"
    done
    exit 0
fi

if [ "$1" == "both" ]; then
    # Print floating points from 0.1 to 0.9 with 0.1 increment
    for i in $(seq 0.05 0.05 0.95); do
        python oselm/train-os-elm.py sample "$DATASET" cuda $i
    done

    for i in $(seq 0.96 0.01 0.99); do
        python oselm/train-os-elm.py sample "$DATASET" cuda $i
    done

    for i in $(seq 2 1 4); do
        for j in $(seq 0.05 0.05 0.95); do
            python oselm/train-os-elm.py batch "$i" "$DATASET" cuda $j
        done

        for j in $(seq 0.96 0.01 0.99); do
            python oselm/train-os-elm.py batch "$i" "$DATASET" cuda $j
        done
    done

    for i in $(seq 5 5 100); do
        for j in $(seq 0.05 0.05 0.95); do
            python oselm/train-os-elm.py batch "$i" "$DATASET" cuda $j
        done

        for j in $(seq 0.96 0.01 0.99); do
            python oselm/train-os-elm.py batch "$i" "$DATASET" cuda $j
        done
    done
    exit 0
fi
