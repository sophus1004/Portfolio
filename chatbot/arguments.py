from dataclasses import dataclass, field
from transformers import HfArgumentParser

@dataclass
class ChatbotArguments:
    chatbot_model_name_or_path: str = field(
        metadata={"help": "The name or path to the chatbot model."}
    )
    rag_embedder_name_or_path: str = field(
        metadata={"help": "The name or path to the chatbot model."}
    )
    rag_collector_name_or_path: str = field(
        metadata={"help": "The name or path to the chatbot model."}
    )
    default_chat_system_prompt: str = field(
        default="당신은 대형 언어 모델인 assistant입니다. user의 질문에 대해 정확하고 유용하며 정보가 풍부한 답변을 제공하는 것이 당신의 역할입니다.",
        metadata={"help": "The name or path to the chatbot model."}
    )
    default_rag_system_prompt: str = field(
        default="당신은 대형 언어 모델인 assistant입니다. user의 질문에 대해 document 내용을 참고해서 정확하게 답변해야 합니다.",
        metadata={"help": "The name or path to the chatbot model."}
    )
    use_system_prompt: bool = field(
        default=True,
        metadata={"help": "Whether to use the system prompt template for fine-tuning or inference."}
    )

def parse_args():
    parser = HfArgumentParser(ChatbotArguments)

    return parser.parse_args_into_dataclasses()