from arguments import chatbot_parse_args

from transformers import AutoModelForCausalLM, AutoTokenizer

import gradio as gr

args = chatbot_parse_args()
chatbot_args = args[0]

model = AutoModelForCausalLM.from_pretrained(chatbot_args.chatbot_model_name_or_path, torch_dtype='auto', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(chatbot_args.chatbot_model_name_or_path)

def respond(applied_system_prompt, message, repetition_penalty, temperature, top_p, history):

    conversation = [{"role": "system", "content": applied_system_prompt}]
    conversation_len = len(history[-5:])

    for idx in range(conversation_len):
        conversation_idx = conversation_len - idx
        conversation.append({"role": "user", "content": history[-conversation_idx][0]})
        conversation.append({"role": "assistant", "content": history[-conversation_idx][1]})
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt"
        ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    response = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(response, skip_special_tokens=True)
    
    history.append((message, response))

    return history, gr.update(value="")


def reset(system_prompt):
    history = []
    if system_prompt == "":
        return history, gr.update(value=default_system_prompt), gr.update(value=default_system_prompt), gr.update(value="")
    
    else:
        return history, gr.update(value=system_prompt), gr.update(value=system_prompt), gr.update(value="")
    
    


def retry(system_prompt, repetition_penalty, temperature, top_p, history):
    if not history:
        return history, gr.update(value="")
    last_message = history[-1][0]
    history.pop()
    return respond(system_prompt, last_message, repetition_penalty, temperature, top_p, history)


def undo(history):
    if not history:
        return history, gr.update(value="")
    history.pop()
    return history, gr.update(value="")


with gr.Blocks(css="./chatbot/gradio_chatbot_style.css") as demo:

    default_system_prompt = "당신은 대형 언어 모델인 assistant입니다. 사용자의 질문에 대해 정확하고 유용하며 정보가 풍부한 답변을 제공하는 것이 당신의 역할입니다."

    with gr.Row(elem_classes="fixed-width"):
        system_prompt = gr.Textbox(label="System Prompt", value=default_system_prompt, interactive=True)
        applied_system_prompt = gr.Textbox(label="Applied system prompt", value=default_system_prompt, interactive=False, visible=False)
        reset_button = gr.ClearButton(value="Reset", elem_classes="fixed-button")

    with gr.Row(elem_classes="fixed-width"):
        chatbot = gr.Chatbot()

    with gr.Row(elem_classes="fixed-width"):
        with gr.Row():
            repetition_penalty = gr.Slider(label="Repetition Penalty", value=1.0, minimum=1.0, maximum=2.0, step=0.1, interactive=True)
        with gr.Row():
            temperature = gr.Slider(label="Temperature", value=1.0, minimum=0.0, maximum=2.0, step=0.1, interactive=True)
        with gr.Row():
            top_p = gr.Slider(label="Top-p", value=1.0, minimum=0.0, maximum=1.0, step=0.1, interactive=True)

    with gr.Row(elem_classes="fixed-width"):
        with gr.Row():
            retry_button = gr.Button("Retry")
        with gr.Row():
            undo_button = gr.Button("Undo")
        with gr.Row():
            gr.Checkbox(label="RAG")

    with gr.Row(elem_classes="fixed-width"):
        message = gr.Textbox(show_label=False, placeholder="Type a message...")
        send_button = gr.Button("Send", elem_classes="fixed-button")

    reset_button.click(reset,
                       inputs=[system_prompt],
                       outputs=[chatbot, applied_system_prompt, system_prompt, message])

    send_button.click(respond, 
                      inputs=[applied_system_prompt, message, repetition_penalty, temperature, top_p, chatbot], 
                      outputs=[chatbot, message])

    retry_button.click(retry, 
                      inputs=[applied_system_prompt, repetition_penalty, temperature, top_p, chatbot], 
                      outputs=[chatbot, message])

    undo_button.click(undo, 
                      inputs=[chatbot], 
                      outputs=[chatbot, message])

if __name__ == "__main__":
    demo.launch(share=True)