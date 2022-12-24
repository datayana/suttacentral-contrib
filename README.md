# suttacentral-contrib

Contrib code to parse and analyze SuttaCentral data


## Create python environment

```bash
conda env create --file conda.yaml
conda activate palibert
```

## Get the raw data

1. Clone the data from SuttaCentral:

    ```bash
    git clone https://github.com/suttacentral/sc-data.git
    ```

2. To make sure you're using the version that has been tested with this code:

    ```
    cd sc-data
    git checkout b7cf33cc624332934514c8db6fb3880ead7d7421
    ```

## Train MLM model "palibert"

```
mkdir data
mkdir models/palibert/
```

```bash
python ./scripts/export_pli_text.py --sc_root_clone ../sc-data --export_train_file ./data/bpe_train_file.txt
```

```bash
python ./scripts/tokenizer_train.py --train_file ./data/bpe_train_file.txt --save_model ./models/palibert/
```

```bash
mkdir models/palibert/
python ./scripts/export_model_config.py --model_name roberta-base --save_config ./models/palibert/
```

```bash
wget https://raw.githubusercontent.com/huggingface/transformers/06886d5a684228a695b29645993b3be55190bd9c/examples/pytorch/language-modeling/run_mlm.py -O scripts/run_mlm.py
```

```
python .\scripts\run_mlm.py `
    --output_dir ./models/palibert/small `
    --model_type roberta-base `
    --config_name ./models/palibert/ `
    --tokenizer_name ./models/palibert/ `
    --max_seq_length 512 `
    --do_train `
    --learning_rate 1e-4 `
    --num_train_epochs 5 `
    --save_total_limit 2 `
    --save_steps 2000 `
    --per_gpu_train_batch_size 16 `
    --seed 42 `
    --train_file .\data\bpe_train_file.txt
```
