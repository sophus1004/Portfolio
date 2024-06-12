class Prompter:
    def __init__(self):
        self.template = "orca"
        self.sysem_prompt = "너는 사용자에게 정보를 제공하는 역할을 한다. 사용자가 질문을 하면, 관련된 사실을 정확하고 간결하게 제공해라. 예를 들어, 역사적 사건, 과학적 원리, 기술적 개념 등에 대해 설명할 수 있다."
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

    def prompt_generator(self, tokenizer, model_template, sysem_prompt):
        if sysem_prompt is None:
            sysem_prompt = self.sysem_prompt

        if model_template not in self.templates:
            response_template = self.templates["base"]['response_template']
            prompt = self.templates["base"]['template']
            prompt = prompt.format(system=sysem_prompt, instruction="{instruction}", output="{output}")

        else:
            response_template = self.templates[model_template]['response_template']
            prompt = tokenizer.apply_chat_template(self.templates[model_template]['template'], tokenize=False)
            prompt = prompt.format(system=sysem_prompt, instruction="{instruction}", output="{output}")

        return prompt, response_template