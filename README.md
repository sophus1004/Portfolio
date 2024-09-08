## What is this Project

이 프로젝트는 개인용 MLOps 제작 프로젝트 입니다. 지원하는 기능은 아래와 같고 지속적으로 기능과 편의성을 업데이트 할 예정입니다.

## Features

### Fine Tuning

- Supervised Fine Tuning (SFT)
- Direct Preference Optimization (DPO)

### Gradio Chatbot

- Multi-Turn Conversation
- Retrieval-Augmented Generation (RAG)

### Utility

- Merging PEFT model

## **Quick Start**

### install

```bash
git clone https://github.com/sophus1004/portfolio.git
cd portfolio
pip install -r requirements.txt
```

### Set Environment

```bash
huggingface-cli login --token {HF_Access_Tokens}
```

### Step 1 - Supervised Fine-Tuning

```bash
bash step1_sft_train.sh
```

**SFT GPU 메모리 사용 가이드**

| Model | Mixed Precision | LoRA | QLoRA |
| --- | --- | --- | --- |
| google/gemma-2-2b-it (1024 tokens) | 58GB | 28GB | 15GB |
| google/gemma-2-2b-it (2048 tokens) | 66GB | 39GB | 25GB |
| google/gemma-2-2b-it (4096 tokens) | 105GB | 78GB | 63GB |
| google/gemma-2-2b-it (8192 tokens) | 250GB | 214GB | - |
| google/gemma-2-9b-it (1024 tokens) | 196GB | 79GB | 30GB |
| google/gemma-2-9b-it (2048 tokens) | 214GB | 109GB | 59GB |
| google/gemma-2-9b-it (4096 tokens) | - | 215GB | 152GB |

**SFT GPU 메모리 사용 가이드 with Gradient checkpointing**

| Model | Mixed Precision | LoRA | QLoRA |
| --- | --- | --- | --- |
| google/gemma-2-2b-it (1024 tokens) | 52GB | 17GB | 8GB |
| google/gemma-2-2b-it (2048 tokens) | 57GB | 22GB | 12GB |
| google/gemma-2-2b-it (4096 tokens) | 66GB | 36GB | 26GB |
| google/gemma-2-2b-it (8192 tokens) | 94GB | 67GB | 58GB |
| google/gemma-2-9b-it (1024 tokens) | 182GB | 44GB | 13GB |
| google/gemma-2-9b-it (2048 tokens) | 190GB | 52GB | 20GB |
| google/gemma-2-9b-it (4096 tokens) | 200GB | 68GB | 35GB |

### Step 2 - Direct Preference Optimization

```bash
bash step2_dpo_train.sh
```

**DPO GPU 메모리 사용 가이드**

| Model | LoRA | QLoRA |
| --- | --- | --- |
| google/gemma-2-2b-it (1024 tokens) | 43GB | 30GB |
| google/gemma-2-2b-it (2048 tokens) | 82GB | 59GB |
| google/gemma-2-2b-it (4096 tokens) | 178GB | - |
| google/gemma-2-9b-it (1024 tokens) | 106GB | 56GB |
| google/gemma-2-9b-it (2048 tokens) | 186GB | 140GB |

**DPO GPU 메모리 사용 가이드 with Gradient checkpointing**

| Model | LoRA | QLoRA |
| --- | --- | --- |
| google/gemma-2-2b-it (1024 tokens) | 27GB | 16GB |
| google/gemma-2-2b-it (2048 tokens) | 41GB | 29GB |
| google/gemma-2-2b-it (4096 tokens) | 71GB | 56GB |
| google/gemma-2-9b-it (1024 tokens) | 54GB | 21GB |
| google/gemma-2-9b-it (2048 tokens) | 70GB | 35GB |
| google/gemma-2-9b-it (4096 tokens) | 110GB | 67GB |

### Gradio Chatbot

```bash
bash chatbot.sh
```

### Utility - Merging PEFT model

```bash
bash utils_merge_lora.sh
```
