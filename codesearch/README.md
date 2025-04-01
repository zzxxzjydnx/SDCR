



## Fine-Tune

```shell
lang=python
mkdir -p ./saved_models/$lang
python run.py \
    --output_dir=./saved_models/python \
    --config_name=../graphcodebert-base \
    --model_name_or_path=../graphcodebert-base \
    --tokenizer_name=../graphcodebert-base \
    --lang=python \
    --do_train \
    --train_data_file=dataset/python/train.jsonl \
    --eval_data_file=dataset/python/valid.jsonl \
    --test_data_file=dataset/python/test.jsonl \
    --codebase_file=dataset/python/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --data_flow_length 64 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1| tee saved_models/python/train.log
    
```
## Inference and Evaluation

```shell
lang=ruby
python run.py \
    --output_dir=saved_models/python \
    --config_name=../graphcodebert-base \
    --model_name_or_path=../graphcodebert-base \
    --tokenizer_name=../graphcodebert-base \
    --lang=python \
    --do_eval \
    --do_test \
    --train_data_file=dataset/python/train.jsonl \
    --eval_data_file=dataset/python/valid.jsonl \
    --test_data_file=dataset/python/test.jsonl \
    --codebase_file=dataset/python/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --data_flow_length 64 \
    --nl_length 128 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 2>&1| tee saved_models/python/test.log
```

