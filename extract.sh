#!/bin/bash

DATASET="$2"
PHASED="$3"
PHASED_FLAG=""

valid_datasets=("fashion-mnist" "mnist" "cifar10" "cifar100" "super-tiny-imagenet" "tiny-imagenet")

valid_phased=("True" "False")

is_valid_phased() {
    for d in "${valid_phased[@]}"; do
        if [ "$d" == "$PHASED" ]; then
            return 0
        fi
    done
    return 1
}

is_valid_dataset() {
    for d in "${valid_datasets[@]}"; do
        if [ "$d" == "$DATASET" ]; then
            return 0
        fi
    done
    return 1
}

if ! is_valid_dataset ; then
    echo "Invalid dataset: $DATASET"
    exit 1
fi

if ! is_valid_phased ; then
    echo "Invalid phased flag: $PHASED_FLAG"
    exit 1
else
    if [ "$PHASED_FLAG" == "True" ]; then
        PHASED_FLAG="--phased"
    fi
fi

# If args is "seq-prop", then run sequential proportion experiment
if [ "$1" == "seq-prop" ]; then
    # Print floating points from 0.1 to 0.9 with 0.1 increment
    for i in $(seq 0.05 0.05 0.95); do
        python oselm/train-os-elm.py --mode sample --dataset "$DATASET" --seq-prop $i $PHASED_FLAG --save-results --result-strategy seq-prop
    done

    for i in $(seq 0.96 0.01 0.99); do
        python oselm/train-os-elm.py --mode sample --dataset "$DATASET" --seq-prop $i $PHASED_FLAG --save-results --result-strategy seq-prop
    done
    exit 0
fi

if [ "$1" == "batch-size" ]; then
    python oselm/train-os-elm.py --mode sample --dataset "$DATASET" $PHASED_FLAG --save-results --result-strategy batch-size
    for i in $(seq 2 1 4); do
        python oselm/train-os-elm.py --mode batch --batch-size "$i" --dataset "$DATASET" $PHASED_FLAG --save-results --result-strategy batch-size
    done

    for i in $(seq 5 5 100); do
        python oselm/train-os-elm.py --mode batch --batch-size "$i" --dataset "$DATASET" $PHASED_FLAG --save-results --result-strategy batch-size
    done
    exit 0
fi

if [ "$1" == "both" ]; then
    # Print floating points from 0.1 to 0.9 with 0.1 increment
    for i in $(seq 0.05 0.05 0.95); do
        python oselm/train-os-elm.py --mode sample --dataset "$DATASET" --seq-prop $i $PHASED_FLAG --save-results --result-strategy total
    done

    for i in $(seq 0.96 0.01 0.99); do
        python oselm/train-os-elm.py --mode sample --dataset "$DATASET" --seq-prop $i $PHASED_FLAG --save-results --result-strategy total
    done

    for i in $(seq 2 1 4); do
        for j in $(seq 0.05 0.05 0.95); do
            python oselm/train-os-elm.py --mode batch --batch-size "$i" --dataset "$DATASET" --seq-prop $j $PHASED_FLAG --save-results --result-strategy total
        done

        for j in $(seq 0.96 0.01 0.99); do
            python oselm/train-os-elm.py --mode batch --batch-size "$i" --dataset "$DATASET" --seq-prop $j $PHASED_FLAG --save-results --result-strategy total
        done
    done

    for i in $(seq 5 5 100); do
        for j in $(seq 0.05 0.05 0.95); do
            python oselm/train-os-elm.py --mode batch --batch-size "$i" --dataset "$DATASET" --seq-prop $j $PHASED_FLAG --save-results --result-strategy total
        done

        for j in $(seq 0.96 0.01 0.99); do
            python oselm/train-os-elm.py --mode batch --batch-size "$i" --dataset "$DATASET" --seq-prop $j $PHASED_FLAG --save-results --result-strategy total
        done
    done
    exit 0
fi
