from dataclasses import dataclass, field
from transformers import HfArgumentParser

@dataclass
class MergeLoraArguments:
    peft_model_name_or_path: str = field(
        metadata={"help": "The name or path to the PEFT model."}
    )
    base_model_name_or_path: str = field(
        metadata={"help": "The name or path to the base pretrained model."}
    )
    output_dir: str = field(
        default=None,
        metadata={"help": "The directory where the merged model will be saved."}
    )

def merge_lora_parse_args():
    parser = HfArgumentParser(MergeLoraArguments)

    return parser.parse_args_into_dataclasses()