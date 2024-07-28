class Prompter:
    def __init__(self):
        self.system_prompt = "당신은 대형 언어 모델인 assistant입니다. 사용자의 질문에 대해 정확하고 유용하며 정보가 풍부한 답변을 제공하는 것이 당신의 역할입니다."
        self.templates = {
            "default":[
                {"role": "system", "content": "{system}"},
                {"role": "user", "content": "{instruction}"},
                {"role": "assistant", "content": "{output}"}
            ],
            "gemma-2":[
                {"role": "user", "content": "{instruction}"},
                {"role": "assistant", "content": "{output}"}
            ]
        }
        self.response_template = {
            "default": "### assistant",
            "llama-3": "<|start_header_id|>assistant<|end_header_id|>",
            "gemma-2": "<start_of_turn>model",
            "qwen2": "<|im_start|>assistant"
        }

    def prompt_generator(self, tokenizer, model_template, system_prompt):
        if tokenizer.chat_template is None:
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if loop.first and messages[0]['role'] != 'system' %}"
                "{{ '### system\nYou are a helpful assistant.\n\n' }}"
                "{% endif %}"
                "{{'### ' + message['role'] + '\n' + message['content'] + '\n\n'}}"
                "{% endfor %}"
                "{% if add_generation_prompt %}{{ 'assistant\n' }}{% endif %}"
            )

        if system_prompt is None:
            system_prompt = self.system_prompt

        template = self.templates.get(model_template, self.templates['default'])
        prompt = tokenizer.apply_chat_template(template, tokenize=False)
        prompt = prompt.format(system=system_prompt, instruction="{instruction}", output="{output}")

        response_template = self.response_template.get(model_template, self.response_template["default"])

        return prompt, response_template