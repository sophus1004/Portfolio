from functools import partial

import json
from arguments import parse_args
from dataclasses import asdict

from rich.console import Console
from rich.panel import Panel
from rich.align import Align
from rich.pretty import Pretty

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig

from datasets import Dataset, load_dataset

from prompter import Prompter

def main():

    args = parse_args()

    training_mode_args = args[0]
    model_args = args[1]
    lora_args = args[2]
    training_args = args[3]

    use_system_prompt = model_args.use_system_prompt

    default_system_prompt = "당신은 대형 언어 모델인 assistant입니다. 사용자의 질문에 대해 정확하고 유용하며 정보가 풍부한 답변을 제공하는 것이 당신의 역할입니다."
    system_prompt = model_args.system_prompt if model_args.system_prompt is not None else default_system_prompt

    model_name_or_path = model_args.model_name_or_path
    data_name_or_path = model_args.data_name_or_path
    train_ds = load_dataset(data_name_or_path)['train']
    train_ds = train_ds.select(range(100))

    use_qlora = training_mode_args.use_qlora
    gradient_checkpointing = training_mode_args.gradient_checkpointing
    training_args = DPOConfig(
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
        loss_type=training_args.loss_type,
        bf16=True if training_args.mixed_precision == 'bf16' else None,
        fp16=True if training_args.mixed_precision == 'fp16' else None,
        max_length=2048,
        max_prompt_length=2048,
        remove_unused_columns=False,
        save_only_model=True
        )
    
    peft_config = LoraConfig(
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
        model.enable_input_require_grads() if use_qlora == True else None
        model.config.use_cache=False
        training_args.gradient_checkpointing=True
        training_args.gradient_checkpointing_kwargs={'use_reentrant':False}
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"]=True
    tokenizer.bos_token = model_args.add_bos_token if model_args.add_bos_token is not None else tokenizer.bos_token

    if tokenizer.bos_token is None:
        print_warning_and_exit()

    prompter = Prompter()
    user_template, assistant_template = prompter.prompt_generator(tokenizer=tokenizer, use_system_prompt=use_system_prompt)

    if use_system_prompt is True:
        user_template = user_template.format(system=system_prompt, instruction="{instruction}", output="{output}")

    train_ds = formatting_prompts_func(train_ds, user_template, assistant_template)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        peft_config=peft_config,
        beta=0.1,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer
        )

    print_args(user_template, assistant_template, training_mode_args, model_args, lora_args, training_args)

    trainer.train()

    print_result()


def formatting_prompts_func(example, user_template, assistant_template):

    format_item = lambda inst, chosen, rejected: {
        'prompt': user_template.format(instruction=inst),
        'chosen': assistant_template.format(output=chosen),
        'rejected': assistant_template.format(output=rejected)
    }

    formatted_data = list(map(
        partial(format_item),
        example['prompt'],
        example['chosen'],
        example['rejected']
    ))

    return Dataset.from_list(formatted_data)


def print_args(user_template, assistant_template, training_mode_args, model_args, lora_args, training_args):
    console = Console()

    user_template_message = Align.left(f"\n{user_template}")
    user_template_panel = Panel(user_template_message, title="User Templates", border_style="bold", width=80)

    assistant_template_message = Align.left(f"\n{assistant_template}")
    assistant_template_panel = Panel(assistant_template_message, title="Assistant Templates", border_style="bold", width=80)

    combined_settings = {
        "Training Mode Args": asdict(training_mode_args),
        "Model Args": asdict(model_args),
        "Lora Args": asdict(lora_args),
        "Training Args": asdict(training_args)
    }

    combined_settings_message = Pretty(combined_settings)
    combined_settings_panel = Panel(combined_settings_message, title="Training Settings", border_style="bold", width=80)

    console.print("\n", user_template_panel, "\n")
    console.print(assistant_template_panel, "\n")
    console.print(combined_settings_panel, "\n")


def print_warning_and_exit():
    console = Console()

    warning_message = Align.center("\n[bold red]Warning: DPO 학습에는 토크나이저에 BOS 토큰이 설정되어 있어야 합니다.")
    console.print("\n", Panel(warning_message, title="Warning", border_style="bold red", width=80, height=5), "\n")

    exit(1)


def print_result():
    console = Console()

    result_message = Align.center("Training has successfully completed.")
    console.print("\n", Panel(result_message, title="Success", border_style="bold", width=50, height=3), "\n")


if __name__ == "__main__":
    main()