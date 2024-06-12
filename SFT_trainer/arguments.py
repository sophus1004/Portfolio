import json

from dataclasses import dataclass, field, asdict
from typing import Optional
from transformers import HfArgumentParser

# Step 1: Define the data class
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
    model_template: str = field(
        metadata={"help": "The prompt template to be used for fine-tuning or inference."}
    )
    sysem_prompt: str = field(
        default=None,
        metadata={"help": "The sysem prompt template to be used for fine-tuning or inference."}
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

# Example usage with HfArgumentParser
parser = HfArgumentParser((ModelArguments, TrainingArguments, TrainingModeArguments, LoRAArguments))

# Parse command line arguments (example_args would be replaced by actual command line args)
example_args = [
    "--model_name_or_path", "bert-base-uncased",
    "--torch_dtype", "float16",
    "--use_lora", "False",
    "--use_qlora", "False",
    "--gradient_checkpointing", "True",
    "--output_dir", "./output",
    "--save_strategy", "epoch",
    "--save_steps", "100",
    "--num_train_epochs", "3",
    "--logging_steps", "10",
    "--per_device_train_batch_size", "16",
    "--gradient_accumulation_steps", "4",
    "--warmup_steps", "100",
    "--learning_rate", "3e-5",
    "--weight_decay", "0.01",
    "--mixed_precision", "fp16",
    "--lora_r", "32",
    "--lora_alpha", "64",
    "--lora_dropout", "0.1",
    "--lora_target_modules", "encoder,decoder"
]

# Create the parser
parser = HfArgumentParser((TrainingModeArguments, ModelArguments, LoRAArguments, TrainingArguments))

def parse_args():
    return parser.parse_args_into_dataclasses()