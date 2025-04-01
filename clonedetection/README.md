# Clone Detection

## Task Definition

Given two codes as the input, the task is to do binary classification (0/1), where 1 stands for semantic equivalence and 0 for others. Models are evaluated by F1 score.

### unizp dataset
unzip dataset


### Fine-tune


```shell
mkdir saved_models
python run.py \
    --output_dir=saved_models \
    --config_name=../graphcodebert-base \
    --model_name_or_path=../graphcodebert-base \
    --tokenizer_name=../graphcodebert-base \
    --do_train \
    --train_data_file=dataset/train.txt \
    --eval_data_file=dataset/valid.txt \
    --test_data_file=dataset/test.txt \
    --epoch 1 \
    --code_length 510 \
    --data_flow_length 128 \
    --train_batch_size 4 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/train.log
```

### Inference

We use full test data for inference. 

```shell
python run.py \
    --output_dir=saved_models \
    --config_name=../graphcodebert-base \
    --model_name_or_path=../graphcodebert-base \
    --tokenizer_name=../graphcodebert-base \
    --do_eval \
    --do_test \
    --train_data_file=dataset/train.txt \
    --eval_data_file=dataset/valid.txt \
    --test_data_file=dataset/test.txt \
    --epoch 1 \
    --code_length 512 \
    --data_flow_length 128 \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/test.log
```

### Evaluation

```shell
python evaluator/evaluator.py -a dataset/test.txt -p saved_models/predictions.txt 2>&1| tee saved_models/score.log
```

