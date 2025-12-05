# RUQuant: Towards Refining Uniform Quantization for Large Language Models

## 📝 Introduction

RUQuant is an efficient quantization framework for large language models that achieves model compression through optimized uniform quantization methods. This project supports low-bit quantization of weights and activations (such as W4A4, W6A6), and provides learnable Householder matrix transformations to further optimize quantization performance.

**📢 This work has been accepted by KDD 2026.**

## 🔧 Installation

### Requirements

- Python 3.10+
- CUDA 11.0+
- PyTorch 2.0+

### Installation Steps

```bash
# Create virtual environment
conda create -n ruquant python=3.10 -y
conda activate ruquant

# Install dependencies
pip install -r requirements.txt
```

## 📂 Project Structure

```
RUQuant/
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies list
├── LICENSE                        # License file
├── main.py                        # Main program entry
├── generate_act_scale_shift.py   # Calculate activation scales and shifts
├── datautils.py                   # Data loading utilities
├── utils.py                       # General utility functions
├── categories.py                  # MMLU category definitions
├── parallel_utils.py              # Multi-GPU parallel utilities
├── act_scales/                    # Pre-computed activation scales
│   ├── llama-7b.pt
│   ├── Llama-2-7b-hf.pt
│   └── ...
├── act_shifts/                    # Pre-computed activation shifts
│   ├── llama-7b.pt
│   ├── Llama-2-7b-hf.pt
│   └── ...
├── quantize/                      # Quantization modules
│   ├── ruquant.py                 # Main RUQuant quantization logic
│   ├── quantizer.py               # Quantizer implementation
│   ├── utils.py                   # Quantization utility functions
│   ├── int_linear.py              # Integer linear layer
│   ├── int_matmul.py              # Integer matrix multiplication
│   └── ...
├── models/                        # Model definitions
│   ├── LMClass.py                 # Language model class
│   ├── int_llama_layer.py         # Quantized LLaMA layers
│   ├── int_mistral_layer.py       # Quantized Mistral layers
│   └── transformation.py          # Householder and Givens rotation transformations
├── lm_eval/                       # Evaluation modules
│   ├── evaluator.py               # Evaluator
│   ├── tasks/                     # Various evaluation tasks
│   └── ...
├── data/                          # Evaluation data
│   └── mt_bench/                  # MT-Bench data
├── run_RUQuant-W4A4.sh            # W4A4 quantization script
├── run_RUQuant-W6A6.sh            # W6A6 quantization script
└── run_RUQuant-W4A4-ft.sh         # W4A4 fine-tuning script (learnable Householder)
```

## 🚀 Quick Start

### Step 1: Generate Activation Scales and Shifts

Before performing quantization, you need to calculate the model's activation scales and shifts for smooth quantization. These factors are fused into the weights without introducing additional parameters or computational overhead.

```bash
python generate_act_scale_shift.py --model PATH_TO_YOUR_MODEL
```

**Arguments:**

- `--model`: Model path (local path or HuggingFace model name)
- `--scales-output-path`: Path to save activation scales (default: `./act_scales/`)
- `--shifts-output-path`: Path to save activation shifts (default: `./act_shifts/`)
- `--calib_dataset`: Calibration dataset (options: wikitext2, ptb, c4, mix, pile)
- `--num-samples`: Number of calibration samples (default: 128)
- `--seq-len`: Sequence length (default: 2048)


### Step 2: Perform Quantization

#### Option 1: Using Predefined Scripts

```bash
# W4A4 quantization (4-bit weight, 4-bit activation)
bash run_RUQuant-W4A4.sh

# W6A6 quantization (6-bit weight, 6-bit activation)
bash run_RUQuant-W6A6.sh

# W4A4 quantization + learnable Householder fine-tuning
bash run_RUQuant-W4A4-ft.sh
```

#### Option 2: Custom Parameters

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model /path/to/your/model \
    --wbits 4 \
    --abits 4 \
    --smooth \
    --lac 0.9 \
    --swc 0.8 \
    --block_size 128 \
    --max_rotation_step 16 \
    --permutation_times 1 \
    --eval_ppl \
    --tasks winogrande,hellaswag,piqa,arc_easy,arc_challenge
```

## 📖 Arguments Explanation

### Basic Arguments

- `--model`: Model path (local path or HuggingFace format)
- `--net`: Model name (for identifying model type)
- `--wbits`: Weight quantization bits (default: 4)
- `--abits`: Activation quantization bits (default: 16)
- `--seed`: Random seed (default: 444)
- `--output_dir`: Log output directory (default: `./log/`)
- `--save_dir`: Directory to save quantized model

### RUQuant-Specific Arguments

- `--block_size`: Block size for rotation matrices, corresponding to hyperparameter **B** in the paper (default: 128)
- `--max_rotation_step`: Maximum rotation steps, corresponding to hyperparameter **K** in the paper (default: 16)
- `--permutation_times`: Number of permutations, corresponding to hyperparameter **T** in the paper (default: 1)
- `--random_permutation_times`: Random permutation times, corresponding to hyperparameter **λ** in the paper (default: 3)

### Quantization Strategy Arguments

- `--smooth`: Enable smooth quantization
- `--lac`: Activation clipping ratio (default: None, range: 0-1)
- `--swc`: Weight clipping ratio (enable without LWC, default: None, range: 0-1)
- `--lwc`: Enable learnable weight clipping
- `--let`: Enable learnable equivalent transformation
- `--lh`: Enable learnable Householder transformation
- `--symmetric`: Use symmetric quantization (default: False)
- `--group_size`: Weight group size (default: None)
- `--act_group_size`: Activation group size (default: None)

## 💡 Usage Examples

### Example 1: LLaMA-7B W4A4 Quantization

```bash
# Step 1: Generate scales and shifts
python generate_act_scale_shift.py --model /path/to/llama-7b

# Step 2: Perform quantization and evaluation
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model /path/to/llama-7b \
    --wbits 4 \
    --abits 4 \
    --smooth \
    --lac 0.9 \
    --swc 0.8 \
    --eval_ppl \
    --tasks winogrande,hellaswag,piqa
```

### Example 2: LLaMA-13B W6A6 Quantization

```bash
# Step 1: Generate scales and shifts
python generate_act_scale_shift.py --model /path/to/llama-13b

# Step 2: Perform quantization
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model /path/to/llama-13b \
    --wbits 6 \
    --abits 6 \
    --smooth \
    --lac 1.0 \
    --swc 1.0 \
    --eval_ppl
```

### Example 3: Using Learnable Householder Fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model /path/to/llama-7b \
    --wbits 4 \
    --abits 4 \
    --epochs 20 \
    --lh \
    --lh_lr 1e-2 \
    --smooth \
    --lac 0.9 \
    --swc 0.8 \
    --eval_ppl
```

## 📊 Experimental Results

### Perplexity on WikiText2 Dataset

The table below shows the perplexity results (↓ lower is better) of LLaMA-1, LLaMA-2, and LLaMA-3 models on the WikiText2 dataset.

| **Bit-width** | **Method**            | **LLaMA-1-7B** | **LLaMA-1-13B** | **LLaMA-1-30B** | **LLaMA-2-7B** | **LLaMA-2-13B** | **LLaMA-3-8B** |
| ------------- | --------------------- | -------------- | --------------- | --------------- | -------------- | --------------- | -------------- |
| **FP16**      | -                     | 5.68           | 5.09            | 4.10            | 5.47           | 4.88            | 6.14           |
| **W4A4**      | SmoothQuant           | 25.25          | 40.05           | 192.40          | 83.12          | 35.88           | 210.19         |
|               | OmniQuant             | 11.26          | 10.87           | 10.33           | 14.26          | 12.30           | 3640.00        |
|               | QLLM                  | 9.65           | 8.41            | 8.37            | 11.75          | 9.09            | -              |
|               | DuQuant               | 6.40           | 5.65            | 4.72            | 6.28           | 5.42            | 8.56           |
|               | **RUQuant**           | **6.29**       | **5.55**        | **4.61**        | **6.17**       | **5.35**        | **8.10**       |
|               | **RUQuant+fine-tune** | **6.14**       | **5.53**        | **4.55**        | **6.04**       | **5.29**        | **7.99**       |
| **W6A6**      | SmoothQuant           | 25.25          | 40.05           | 192.40          | 83.12          | 35.88           | 7.07           |
|               | OmniQuant             | 11.26          | 10.87           | 10.33           | 14.26          | 12.30           | 7.24           |
|               | QLLM                  | 9.65           | 8.41            | 8.37            | 11.75          | 9.09            | -              |
|               | DuQuant               | 5.73           | 5.13            | 4.14            | 5.53           | 4.92            | 6.27           |
|               | **RUQuant**           | **5.71**       | **5.12**        | **4.13**        | **5.51**       | **4.91**        | **6.25**       |



## 🔗 Related Resources

- [Duquant](https://github.com/Hsu1023/Duquant)
- [OmniQuant](https://github.com/OpenGVLab/OmniQuant)

## 🙏 Acknowledgements

This repository is built upon the following projects:

- [Duquant](https://github.com/Hsu1023/Duquant)
- [OmniQuant](https://github.com/OpenGVLab/OmniQuant)

We thank the authors for their outstanding work!