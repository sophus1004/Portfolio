#Chatbot 실행

python chatbot/gradio_chatbot.py \
   --chatbot_model_name_or_path google/gemma-2-9b-it \
   --rag_embedder_name_or_path BAAI/bge-m3 \
   --rag_collector_name_or_path CHOJW1004/maywell_ko_wikidata_QA_12800 \
   --use_system_prompt True \