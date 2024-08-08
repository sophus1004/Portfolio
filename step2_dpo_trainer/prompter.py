class Prompter:
    def __init__(self):
        self.system_prompt = "당신은 대형 언어 모델인 assistant입니다. 사용자의 질문에 대해 정확하고 유용하며 정보가 풍부한 답변을 제공하는 것이 당신의 역할입니다."
        self.templates = {
            "system":[
                {"role": "system", "content": "{system}"}
            ],
            "user":[
                {"role": "system", "content": "{system}"},
                {"role": "user", "content": "{instruction}"}
            ],
            "assistant":[
                {"role": "system", "content": "{system}"},
                {"role": "assistant", "content": "{output}"}
            ]
        }

    def prompt_generator(self, tokenizer, use_system_prompt):
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

        if use_system_prompt is True:
            system_template = tokenizer.apply_chat_template(self.templates["system"], tokenize=False)
            user_template = tokenizer.apply_chat_template(self.templates["user"], tokenize=False)
            assistant_template = tokenizer.apply_chat_template(self.templates["assistant"], tokenize=False).replace(system_template, "")

        else:
            user_template = tokenizer.apply_chat_template([item for item in self.templates["user"] if item["role"] == "user"], tokenize=False)
            assistant_template = tokenizer.apply_chat_template([item for item in self.templates["assistant"] if item["role"] == "assistant"], tokenize=False)

        return user_template, assistant_template