#SFT 실행

python chatbot/gradio_chatbot.py \
   --chatbot_model_name_or_path Qwen/Qwen2-1.5B-Instruct \
   --rag_embedder_name_or_path jhgan/ko-sroberta-multitask \
   --rag_collector_name_or_path CHOJW1004/maywell_ko_wikidata_QA_12800 