#!/bin/bash
# Train 10 neural networks with different random initializations and average the results when making predictions
# args are model directory, number of training steps, tfrecord directory, S/N threshold

dir=$1
steps=$2
TFRECORD_DIR=$3

if ! [ -d "$dir" ]; then
	mkdir "$dir"
fi

for i in {1..10}
do
	echo "Training model $i"
	model_dir="$dir/model_$i"
	mkdir $model_dir
	python astronet/train.py \
		--model=AstroCNNModel \
		--config_name=local_global \
		--train_files=$TFRECORD_DIR/train* \
		--eval_files=$TFRECORD_DIR/val* \
		--model_dir=$model_dir \
		--train_steps=$steps \

done

