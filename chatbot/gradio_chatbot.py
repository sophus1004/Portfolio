import gradio as gr
from arguments import parse_args

from chatbot_functions import ChatbotFunctions


def main():

    args = parse_args()

    ChatbotArguments = args[0]

    chatbot_model_name_or_path = ChatbotArguments.chatbot_model_name_or_path
    rag_embedder_name_or_path = ChatbotArguments.rag_embedder_name_or_path
    rag_collector_name_or_path = ChatbotArguments.rag_collector_name_or_path
    default_chat_system_prompt = ChatbotArguments.default_chat_system_prompt
    default_rag_system_prompt = ChatbotArguments.default_rag_system_prompt

    chatbot = ChatbotFunctions(chatbot_model_name_or_path,
                               rag_embedder_name_or_path,
                               rag_collector_name_or_path,
                               default_chat_system_prompt,
                               default_rag_system_prompt
                               )

    with gr.Blocks() as demo:
        with gr.Tab("Multiturn Chatbot"):
            with gr.Row():
                chat_system_prompt = gr.Textbox(label="System Prompt", value=default_chat_system_prompt, interactive=True, min_width=800)
                chat_applied_system_prompt = gr.Textbox(label="Applied system prompt", value=default_chat_system_prompt, interactive=False, visible=False)
                chat_reset_button = gr.ClearButton(value="Reset")

            with gr.Row():
                chat_chatbot = gr.Chatbot()

            with gr.Row():
                with gr.Row():
                    chat_repetition_penalty = gr.Slider(label="Repetition Penalty", value=1.0, minimum=1.0, maximum=2.0, step=0.1, interactive=True)
                with gr.Row():
                    chat_temperature = gr.Slider(label="Temperature", value=1.0, minimum=0.0, maximum=2.0, step=0.1, interactive=True)
                with gr.Row():
                    chat_top_p = gr.Slider(label="Top-p", value=1.0, minimum=0.0, maximum=1.0, step=0.05, interactive=True)

            with gr.Row():
                with gr.Row():
                    chat_retry_button = gr.Button("Retry")
                with gr.Row():
                    chat_undo_button = gr.Button("Undo")
                with gr.Row():
                    gr.Checkbox(label="RAG")

            with gr.Row():
                chat_message = gr.Textbox(show_label=False, placeholder="Type a message...", min_width=1000)
                chat_send_button = gr.Button(value="Send")

            chat_reset_button.click(chatbot.chat_reset,
                            inputs=[chat_system_prompt],
                            outputs=[chat_chatbot, chat_applied_system_prompt, chat_system_prompt, chat_message])

            chat_retry_button.click(chatbot.retry, 
                            inputs=[chat_applied_system_prompt, chat_repetition_penalty, chat_temperature, chat_top_p, chat_chatbot], 
                            outputs=[chat_chatbot, chat_message])

            chat_undo_button.click(chatbot.undo, 
                            inputs=[chat_chatbot], 
                            outputs=[chat_chatbot, chat_message])
            
            chat_send_button.click(chatbot.chat_respond, 
                            inputs=[chat_applied_system_prompt, chat_message, chat_repetition_penalty, chat_temperature, chat_top_p, chat_chatbot], 
                            outputs=[chat_chatbot, chat_message])

        with gr.Tab("RAG"):
            gr.Markdown("# Welcome to the Greeting App")
            with gr.Row():
                rag_system_prompt = gr.Textbox(label="System Prompt", value=default_rag_system_prompt, interactive=True, min_width=800)
                rag_applied_system_prompt = gr.Textbox(label="Applied system prompt", value=default_rag_system_prompt, interactive=False, visible=False)
                rag_reset_button = gr.ClearButton(value="Reset")

            with gr.Row():
                with gr.Row():
                    rag_doc_1 = gr.Textbox(label="Document 1", lines=7)
                with gr.Row():
                    rag_doc_2 = gr.Textbox(label="Document 2", lines=7)
                with gr.Row():
                    rag_doc_3 = gr.Textbox(label="Document 3", lines=7)

            with gr.Row():
                rag_input = gr.Textbox(label="Search box", placeholder="Enter your questions...", min_width=750)
                with gr.Row():
                    rag_selected_doc = gr.Radio(label="Using document", choices=["Document 1", "Document 2", "Document 3"])

            with gr.Row():
                with gr.Row():
                    rag_repetition_penalty = gr.Slider(label="Repetition Penalty", value=1.0, minimum=1.0, maximum=2.0, step=0.1, interactive=True)
                with gr.Row():
                    rag_temperature = gr.Slider(label="Temperature", value=1.0, minimum=0.0, maximum=2.0, step=0.1, interactive=True)
                with gr.Row():
                    rag_top_p = gr.Slider(label="Top-p", value=1.0, minimum=0.0, maximum=1.0, step=0.1, interactive=True)
                with gr.Row():
                    rag_inference_button = gr.Button(value="Inference")

            with gr.Row():
                rag_inference_output = gr.Textbox(label="Result")

            rag_reset_button.click(chatbot.rag_reset,
                            inputs=[rag_system_prompt],
                            outputs=[rag_system_prompt, rag_applied_system_prompt, rag_inference_output])

            rag_input.change(chatbot.retrieval, inputs=rag_input, outputs=[rag_doc_1, rag_doc_2, rag_doc_3])
            rag_inference_button.click(chatbot.rag_inference, 
                            inputs=[rag_applied_system_prompt, rag_input, rag_repetition_penalty, rag_temperature, rag_top_p, rag_selected_doc, rag_doc_1, rag_doc_2, rag_doc_3],
                            outputs=rag_inference_output)

    demo.launch(share=True)


if __name__ == "__main__":
    main()