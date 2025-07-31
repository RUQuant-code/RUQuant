# RUQuant: Towards Refining Uniform Quantization for Large Language Models

##  Installation
```bash
conda create -n ruquant python=3.10 -y
conda activate ruquant
pip install -r requirements.txt
```

# Explanation of key documents

```
RUQuant/
├── README.md               	# Project Description Document
├── requirements.txt       	 	# Python dependency list
├── generate_act_scale_shift.py # Calculate the scaling factor of the model
├── main.py                 	# Main program entry
├──act_scales    				# Scaling factor for smoothing
├──act_shifts               	# Offset factor for smoothing
├──cache                    	# The cache of the calibration dataset and the test dataset will be automatically downloaded from huggingface if it is empty
├── quantize/
│   ├── ruquant.py          	# Execute ruquant
│   └── quantize.py         	# The main functions of quantization, including Householder reflection and Givens rotation
├──run_RUQuant-W4A4.sh          # Script for performing W4A4 quantization on models
├──run_RUQuant-W6A6.sh          # Script for performing W6A6 quantization on models
├──run_RUQuant-W4A4-ft.sh       # Script for Fine-tuning the learnable householder matrix
```

## Usage

### 1. Calculate the scaling factor of the model

Before performing ruquant, we need to smooth the model so that the scaling factor can be incorporated into the weights and scaling without introducing any additional parameters or computational overhead.

```bash
python generate_act_scale_shift.py --model PATH_OF_MODEL 
```

### 2. ruquant
```bash
#Perform W4A4 quantification on the model
bash run_RUQuant-W4A4.sh 

#Perform W6A6 quantification on the model
bash run_RUQuant-W6A6.sh 

#Fine-tuning the learnable householder matrix
bash run_RUQuant-W4A4-ft.sh
```


#### Explanation of arguments:
- `--model`: the local model path or huggingface format.
- `--wbits`: weight quantization bits.
- `--abits`: activation quantization bits.
- `--block_size`: the block size of rotation matrices(The hyperparameter $B$ mentioned in the article.).
- `--max_rotation_step`: The hyperparameter $K$ mentioned in the article. 
- `--permutation_times`: The hyperparameter $T$ mentioned in the article..
- `--swc`: the ratio of weight clipping (enable without LWC operation).
- `--lac`: the ratio of activation clipping.
- `--epochs`: the training epochs of LH.
- `--lh`: activate learnable equivalent transformation.
- `--lh_lr`: the learning rate of LH.
- `--resume`: loading pre-trained RUQuant parameters.
- `--multigpu`: to inference larger network on multiple GPUs.
- `--save_dir`: saving the quantization model for further exploration.
- `--eval_ppl`: evaluating the perplexity of quantized models.
- `--tasks`: evaluating on the zero-shot tasks.
- `--eval_mmlu`: evaluating on the MMLU benchmarks.
- `--mmlu_data_dir`: data path of the MMLU benchmarks.
- `--eval_mtbench`: evaluating on the MT-Bench.

## Acknowledgement
This repo is built upon the following projects:

* [DuQuant](https://github.com/Hsu1023/DuQuant)
* [OmniQuant](https://github.com/OpenGVLab/OmniQuant)

We thank the authors for their code.

