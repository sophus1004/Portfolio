class Prompter:
    def __init__(self):
        self.template = "orca"
        self.system_prompt = "당신은 대형 언어 모델인 assistant입니다. 사용자의 질문에 대해 정확하고 유용하며 정보가 풍부한 답변을 제공하는 것이 당신의 역할입니다."
        self.templates = {
            "base":{
                "response_template": "### assistant",
                "template": "### system\n{system}\n\n### user\n{instruction}\n\n### assistant\n{output}"
            },
            "llama-3":{
                "response_template": "<|start_header_id|>assistant<|end_header_id|>",
                "template": [
                    {"role": "system", "content": "{system}"},
                    {"role": "user", "content": "{instruction}"},
                    {"role": "assistant", "content": "{output}"}
                    ]
            },
            "qwen2":{
                "response_template": "<|im_start|>assistant",
                "template": [
                    {"role": "system", "content": "{system}"},
                    {"role": "user", "content": "{instruction}"},
                    {"role": "assistant", "content": "{output}"}
                    ]
            }
        }

    def prompt_generator(self, tokenizer, model_template, system_prompt):
        if system_prompt is None:
            system_prompt = self.system_prompt

        if model_template not in self.templates:
            response_template = self.templates["base"]['response_template']
            prompt = self.templates["base"]['template']
            prompt = prompt.format(system=system_prompt, instruction="{instruction}", output="{output}")

        else:
            response_template = self.templates[model_template]['response_template']
            prompt = tokenizer.apply_chat_template(self.templates[model_template]['template'], tokenize=False)
            prompt = prompt.format(system=system_prompt, instruction="{instruction}", output="{output}")

        return prompt, response_template