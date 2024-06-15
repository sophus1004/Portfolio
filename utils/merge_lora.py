from arguments import merge_lora_parse_args

from rich.console import Console
from rich.panel import Panel
from rich.align import Align

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

def main():
    args = merge_lora_parse_args()
    merge_lora_args = args[0]

    peft_model_name_or_path = merge_lora_args.peft_model_name_or_path
    base_model_name_or_path = merge_lora_args.base_model_name_or_path
    output_dir = merge_lora_args.output_dir

    config = AutoConfig.from_pretrained(base_model_name_or_path)
    save_dtype = str(config.torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=torch.float32, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)

    model = PeftModel.from_pretrained(model,peft_model_name_or_path)
    model = model.merge_and_unload()

    if save_dtype == "torch.float16":
        model.half()

    if save_dtype == "torch.bfloat16":
        model = model.to(torch.bfloat16)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    main()

    console = Console()
    message = Align.center("Merging has successfully completed.")
    console.print("\n", Panel(message, title="Success", border_style="bold", width=50, height=3), "\n")