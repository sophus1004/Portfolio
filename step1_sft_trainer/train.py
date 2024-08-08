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
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

from datasets import load_dataset

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
    attn_implementation = model_args.attention_implementation
    train_ds = load_dataset(data_name_or_path)['train']

    use_lora = training_mode_args.use_lora
    use_qlora = training_mode_args.use_qlora
    gradient_checkpointing = training_mode_args.gradient_checkpointing
    training_args = SFTConfig(
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
        max_seq_length=2048,
        save_only_model=True
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
        attn_implementation=attn_implementation,
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
    generated_prompt, response_template= prompter.prompt_generator(tokenizer=tokenizer, use_system_prompt=use_system_prompt)
    generated_prompt = generated_prompt.format(system=system_prompt, instruction="{instruction}", output="{output}")

    if use_lora == True or use_qlora == True:
        model = get_peft_model(model, lora_config)

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer, mlm=False
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        formatting_func=lambda example: formatting_prompts_func(example, generated_prompt=generated_prompt),
        data_collator=data_collator
        )
    
    print_args(generated_prompt, training_mode_args, model_args, lora_args, training_args)

    trainer.train()

    print_result()


def formatting_prompts_func(example, generated_prompt):
    output_texts = list(map(lambda inst, out: generated_prompt.format(instruction=inst, output=out), 
                            example['instruction'], 
                            example['output']))
    return output_texts


def print_args(generated_prompt, training_mode_args, model_args, lora_args, training_args):
    console = Console()

    input_prompt_message = Align.left(f"\n{generated_prompt}")
    input_prompt_panel = Panel(input_prompt_message, title="Input Prompt", border_style="bold", width=80)

    combined_settings = {
        "Training Mode Args": asdict(training_mode_args),
        "Model Args": asdict(model_args),
        "Lora Args": asdict(lora_args),
        "Training Args": asdict(training_args)
    }

    if training_mode_args.use_lora == False and training_mode_args.use_qlora == False:
        del combined_settings["Lora Args"]

    combined_settings_message = Pretty(combined_settings)
    combined_settings_panel = Panel(combined_settings_message, title="Training Settings", border_style="bold", width=80)

    console.print("\n", input_prompt_panel, "\n")
    console.print(combined_settings_panel, "\n")


def print_result():
    console = Console()

    result_message = Align.center("Training has successfully completed.")
    console.print("\n", Panel(result_message, title="Success", border_style="bold", width=50, height=3), "\n")


if __name__ == "__main__":
    main()