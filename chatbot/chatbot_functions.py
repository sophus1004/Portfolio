import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

import gradio as gr

class ChatbotFunctions:
    def __init__(self, chatbot_model_name_or_path, rag_embedder_name_or_path, rag_collector_name_or_path, default_chat_system_prompt, default_rag_system_prompt):
        self.default_chat_system_prompt = default_chat_system_prompt
        self.default_rag_system_prompt = default_rag_system_prompt

        self.model = AutoModelForCausalLM.from_pretrained(chatbot_model_name_or_path, torch_dtype='auto', device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(chatbot_model_name_or_path)
        self.embedder = SentenceTransformer(rag_embedder_name_or_path)
        self.collector = load_dataset(rag_collector_name_or_path)
        self.corpus_embeddings = self.embedder.encode(self.collector['train']['output'], convert_to_tensor=True)

        self.max_new_tokens = 512
    
    def chat_respond(self, applied_system_prompt, message, repetition_penalty, temperature, top_p, history):
        conversation = [{"role": "system", "content": applied_system_prompt}]
        conversation_len = len(history[-5:])

        for idx in range(conversation_len):
            conversation_idx = conversation_len - idx
            conversation.append({"role": "user", "content": history[-conversation_idx][0]})
            conversation.append({"role": "assistant", "content": history[-conversation_idx][1]})
        conversation.append({"role": "user", "content": message})

        input_ids = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
            ).to(self.model.device)

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=float(repetition_penalty),
            do_sample=True,
            temperature=float(temperature),
            top_p=top_p
            )

        response = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(response, skip_special_tokens=True)
        history.append((message, response))
    
        for item in conversation:
            print(item)
        print("")
        print("--------------------------")
        print("")

        return history, gr.update(value="")

    def retrieval(self, name):
        queries = [name]

        top_k = 3
        for query in queries:
            query_embedding = self.embedder.encode(query, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(query_embedding, self.corpus_embeddings)[0]
            cos_scores = cos_scores.cpu()
            top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

        result_1 = self.collector['train']['output'][top_results[0]].strip()
        result_2 = self.collector['train']['output'][top_results[1]].strip()
        result_3 = self.collector['train']['output'][top_results[2]].strip()

        return result_1, result_2, result_3

    def rag_inference(self, applied_system_prompt, rag_input, repetition_penalty, temperature, top_p, rag_selected_doc, rag_doc_1, rag_doc_2, rag_doc_3):
        documents = {
            "Document 1": rag_doc_1,
            "Document 2": rag_doc_2,
            "Document 3": rag_doc_3
        }
        
        conversation = [{"role": "system", "content": applied_system_prompt},
                        {"role": "document", "content": documents[rag_selected_doc]},
                        {"role": "user", "content": rag_input}]

        input_ids = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt"
            ).to(self.model.device)
        
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=float(repetition_penalty),
            do_sample=True,
            temperature=float(temperature),
            top_p=top_p
            )

        response = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(response, skip_special_tokens=True)

        for item in conversation:
            print(item)
        print("")
        print("--------------------------")
        print("")
        
        return response

    def chat_reset(self, system_prompt):
        history = []
        if system_prompt == "":
            return history, gr.update(value=self.default_chat_system_prompt), gr.update(value=self.default_chat_system_prompt), gr.update(value="")
        
        else:
            return history, gr.update(value=system_prompt), gr.update(value=system_prompt), gr.update(value="")

    def rag_reset(self, system_prompt):
        if system_prompt == "":
            return gr.update(value=self.default_rag_system_prompt), gr.update(value=self.default_rag_system_prompt), gr.update(value="")
        
        else:
            return gr.update(value=system_prompt), gr.update(value=system_prompt), gr.update(value="")
        
    def retry(self, system_prompt, repetition_penalty, temperature, top_p, history):
        if not history:
            return history, gr.update(value="")
        last_message = history[-1][0]
        history.pop()
        return self.chat_respond(system_prompt, last_message, repetition_penalty, temperature, top_p, history)

    def undo(self, history):
        if not history:
            return history, gr.update(value="")
        history.pop()
        return history, gr.update(value="")