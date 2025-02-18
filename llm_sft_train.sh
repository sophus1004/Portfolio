# SFT 실행

# gemma-2 train set
python llm_sft_trainer/train.py \
   --use_lora True \
   --use_qlora False \
   --gradient_checkpointing True \
   --data_name_or_path maywell/ko_wikidata_QA \
   --model_name_or_path google/gemma-2-2b-it \
   --attention_implementation eager \
   --use_system_prompt False \
   --output_dir ./storage/trained_sft_models \
   --save_strategy epoch \
   --num_train_epochs 1 \
   --logging_steps 1 \
   --per_device_train_batch_size 1 \
   --gradient_accumulation_steps 1 \
   --warmup_steps 0 \
   --learning_rate 1e-5 \
   --weight_decay 0.0 \
   --mixed_precision bf16


# Llama-3.1 train set
# python llm_sft_trainer/train.py \
#    --use_lora True \
#    --use_qlora False \
#    --gradient_checkpointing False \
#    --data_name_or_path maywell/ko_wikidata_QA \
#    --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
#    --output_dir ./storage/trained_sft_models \
#    --save_strategy epoch \
#    --num_train_epochs 1 \
#    --logging_steps 1 \
#    --per_device_train_batch_size 1 \
#    --gradient_accumulation_steps 1 \
#    --warmup_steps 0 \
#    --learning_rate 1e-5 \
#    --weight_decay 0.0 \
#    --mixed_precision bf16