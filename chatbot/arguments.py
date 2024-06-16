from dataclasses import dataclass, field
from transformers import HfArgumentParser

@dataclass
class ChatbotArguments:
    chatbot_model_name_or_path: str = field(
        metadata={"help": "The name or path to the chatbot model."}
    )

def chatbot_parse_args():
    parser = HfArgumentParser(ChatbotArguments)

    return parser.parse_args_into_dataclasses()