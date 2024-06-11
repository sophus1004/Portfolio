def SelectPrompt(prompt_name):
    prompts = {
        "llama3_prompt": {
            "system_prompt_no_doc": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n당신은 지식이 풍부한 정보 제공자입니다. 사용자가 질문하는 주제에 대해 정확하고 자세한 정보를 제공하세요.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
            "assistant_prompt": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            }
        }
    
    return prompts[prompt_name]