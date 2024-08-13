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

### Step 2 - Direct Preference Optimization

```bash
bash step2_dpo_train.sh
```

### Utility - Merging PEFT model

```bash
bash utils_merge_lora.sh
```

### Gradio Chatbot

```bash
bash chatbot.sh
```