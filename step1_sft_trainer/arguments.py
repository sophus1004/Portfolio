from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser

@dataclass
class TrainingModeArguments:
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA (Low-Rank Adaptation)."}
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "Whether to use QLoRA (Quantized LoRA)."}
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether to use gradient checkpointing to save memory."}
    )

@dataclass
class ModelArguments:
    data_name_or_path: str = field(
        metadata={"help": "The name or path to the pretraining data."}
    )
    model_name_or_path: str = field(
        metadata={"help": "The name or path to the pretrained model."}
    )
    system_prompt: str = field(
        default=None,
        metadata={"help": "The system_prompt prompt template to be used for fine-tuning or inference."}
    )
    use_system_prompt: bool = field(
        default=True,
        metadata={"help": "Whether to use the system prompt template for fine-tuning or inference."}
    )

@dataclass
class TrainingArguments:
    output_dir: str = field(
        default="./trained_models",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    save_strategy: str = field(
        default="epoch",
        metadata={"help": "The strategy to use for saving checkpoints (e.g., 'epoch', 'steps')."}
    )
    save_steps : int = field(
        default=500,
        metadata={"help": "Save checkpoint every X updates steps."}
    )
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "Total number of training epochs to perform."}
    )
    logging_steps: int = field(
        default=1,
        metadata={"help": "Log every X updates steps."}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    warmup_steps: int = field(
        default=1,
        metadata={"help": "Linear warmup over warmup_steps."}
    )
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "The initial learning rate for AdamW optimizer."}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay to apply (if any)."}
    )
    mixed_precision: Optional[str] = field(
        default="bf16",
        metadata={"help": "Whether to use mixed precision training ('fp16', 'bf16'). Default is None."}
    )

@dataclass
class LoRAArguments:
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA rank (the rank of the low-rank matrices)."}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha (scaling factor for the low-rank matrices)."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "LoRA dropout rate (probability of setting a weight to zero)."}
    )
    lora_target_modules: Optional[str] = field(
        default="all-linear",
        metadata={"help": "Target modules for LoRA. If None, apply LoRA to all layers. Default is 'all-linear'."}
    )

def parse_args():
    parser = HfArgumentParser((TrainingModeArguments, ModelArguments, LoRAArguments, TrainingArguments))

    return parser.parse_args_into_dataclasses()