# Merge LoRA

python utils/merge_lora.py \
   --peft_model_name_or_path storage/sample_lora \
   --base_model_name_or_path Qwen/Qwen2-1.5B-Instruct \
   --output_dir storage/merged_lora_model \