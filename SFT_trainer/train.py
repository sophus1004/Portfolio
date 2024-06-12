import json
from arguments import parse_args
from dataclasses import dataclass, field, asdict

import torch

from transformers import HfArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from datasets import load_dataset

from prompter import Prompter

def main():

    args = parse_args()

    training_mode_args = args[0]
    model_args = args[1]
    lora_args = args[2]
    training_args = args[3]

    model_template = model_args.model_template
    sysem_prompt = model_args.sysem_prompt

    model_name_or_path = model_args.model_name_or_path
    data_name_or_path = model_args.data_name_or_path
    train_ds = load_dataset(data_name_or_path)['train']

    use_lora = training_mode_args.use_lora
    use_qlora = training_mode_args.use_qlora
    gradient_checkpointing=training_mode_args.gradient_checkpointing
    input_training_args = TrainingArguments(
        output_dir=training_args.output_dir,
        save_strategy=training_args.save_strategy,
        save_steps=training_args.save_steps,
        num_train_epochs=training_args.num_train_epochs,
        logging_steps=training_args.logging_steps,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        warmup_steps=training_args.warmup_steps,
        optim="adamw_torch" if not use_qlora == True else "paged_adamw_8bit",
        learning_rate=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        lr_scheduler_type="cosine",
        bf16=True if training_args.mixed_precision == 'bf16' else None,
        fp16=True if training_args.mixed_precision == 'fp16' else None,
        )
    
    if use_lora == True or use_qlora == True:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
            )

    if use_qlora == True:
        bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32 if not use_qlora == True else 'auto',
        quantization_config=bnb_config if use_qlora == True else None,
        device_map='auto'
        )

    if gradient_checkpointing is True:
        model.enable_input_require_grads() if use_lora == True or use_qlora == True else None
        model.config.use_cache=False
        training_args.gradient_checkpointing=True
        training_args.gradient_checkpointing_kwargs={'use_reentrant':False}
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"]=True

    prompter = Prompter()
    generated_prompt, response_template= prompter.prompt_generator(tokenizer=tokenizer, model_template=model_template, sysem_prompt=sysem_prompt)

    if use_lora == True or use_qlora == True:
        model = get_peft_model(model, lora_config)

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer, mlm=False
        )

    trainer = SFTTrainer(
        model=model,
        args=input_training_args,
        train_dataset=train_ds,
        max_seq_length=2048,
        formatting_func=lambda example: formatting_prompts_func(example, generated_prompt=generated_prompt),
        data_collator=data_collator
        )
    print("-- 입력 프롬프트 ----------\n\n", generated_prompt, "\n\n-- 학습 설정 ----------")
    print(json.dumps(asdict(training_mode_args), indent=0)[1:-1])
    print(json.dumps(asdict(model_args), indent=0)[1:-1])
    print(json.dumps(asdict(lora_args), indent=0)[1:-1])
    print(json.dumps(asdict(training_args), indent=0)[1:-1], "\n")

    trainer.train()

def formatting_prompts_func(example, generated_prompt):

    output_texts = list(map(lambda inst, out: generated_prompt.format(instruction=inst, output=out), 
                            example['instruction'], 
                            example['output']))
    return output_texts


if __name__ == "__main__":
    main()