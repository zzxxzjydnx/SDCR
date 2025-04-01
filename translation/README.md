# Code Translation

## Task Definition

Code translation aims to migrate legacy software from one programming language in a platform toanother.
Given a piece of Java (C#) code, the task is to translate the code into C# (Java) version. 
Models are evaluated by BLEU scores and accuracy (exactly match).

### unizp dataset
unzip data


### Fine-tune

```shell
source=java
target=cs
lr=1e-4
batch_size=32
beam_size=10
source_length=320
target_length=256
output_dir=saved_models/$source-$target/
train_file=data/train.java-cs.txt.$source,data/train.java-cs.txt.$target
dev_file=data/valid.java-cs.txt.$source,data/valid.java-cs.txt.$target
epochs=100
pretrained_model=../graphcodebert-base

mkdir -p $output_dir
python run.py \
--do_train \
--do_eval \
--model_type roberta \
--source_lang $source \
--model_name_or_path $pretrained_model \
--tokenizer_name ../graphcodebert-base \
--config_name ../graphcodebert-base \
--train_filename $train_file \
--dev_filename $dev_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--num_train_epochs $epochs 2>&1| tee $output_dir/train.log
```

### Inference

We use full test data for inference. 

```shell
batch_size=64
dev_file=data/valid.java-cs.txt.$source,data/valid.java-cs.txt.$target
test_file=data/test.java-cs.txt.$source,data/test.java-cs.txt.$target
load_model_path=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

python run.py \
--do_test \
--model_type roberta \
--source_lang $source \
--model_name_or_path $pretrained_model \
--tokenizer_name ../graphcodebert-base \
--config_name ../graphcodebert-base \
--load_model_path $load_model_path \
--dev_filename $dev_file \
--test_filename $test_file \
--output_dir $output_dir \
--max_source_length $source_length \
--max_target_length $target_length \
--beam_size $beam_size \
--eval_batch_size $batch_size 2>&1| tee $output_dir/test.log
```



