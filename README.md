# suttacentral-contrib
Contrib code to parse and analyze SuttaCentral data


## Create python environment

```bash
conda env create --file conda_cpu.yaml
conda activate palibertcpu
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

### 1. Create initial training dataset

Create an initial training dataset made of all pali texts in the suttas:

```bash
python ./scripts/export_pli_text.py --sc_root_clone ../sc-data --export_train_file ./data/bpe_train_file.txt
```

### 2. Create model config

Create an initial configuration for a language model "palibert-base". This will train a BPE tokenizer first, and use the parameters to create the model config architecture.

```bash
python ./scripts/create_palibert_base_config.py --output_dir ./models/palibert-base/config --train_file ./data/bpe_train_file.txt
```

### 3.a. Train locally

If you have capacity to run this locally, go ahead and run:

```
python ./scripts/run_mlm.py `
    --output_dir ./models/palibert-base/v1/ `
    --config_name ./models/palibert-base/config/ `
    --tokenizer_name ./models/palibert-base/config/ `
    --max_seq_length 512 `
    --do_train `
    --learning_rate 1e-4 `
    --num_train_epochs 5 `
    --save_total_limit 2 `
    --save_steps 2000 `
    --per_gpu_train_batch_size 16 `
    --seed 42 `
    --train_file ./data/bpe_train_file.txt
```

### 3.b. Train in the cloud (Azure ML)

Alternatively, you can run this job in AzureML using [CLI v2](https://learn.microsoft.com/en-us/azure/machine-learning/concept-v2#azure-machine-learning-cli-v2).

```bash
# create environment
az ml environment create --file ./cloud/azureml/environments/env.yaml

# upload training dataset
az ml data create --file ./cloud/azureml/data/bpe_train.yml

# train from scratch
az ml job create --file ./cloud/azureml/jobs/train_palibert_from_scratch.yml --set compute=COMPUTENAME
```
