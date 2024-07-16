def main_md():
    markdown = """# Customized LLM Chatbot Cho's Portfolio

이 챗봇은 제 개인 테스트와 포트폴리오 전용 챗봇입니다. 앞으로 LLM 발전에 맞춰 지속적으로 업데이트 할 예정입니다."""
    
    return markdown

def multiturn_chatbot_md():
    markdown = """## Multiturn Chatbot

**고정 파라미터**

- **Max Multi-turn :** 5 turn
- **Max new tokens :** 512 token

**기능**

- **Reset :** 모든 챗 기록을 삭제하고 입력된 System Prompt 을 적용
- **Retry :** 새로운 답변 생성
- **Undo :** 이전 턴으로 돌아가기"""

    return markdown

def RAG_md():
    markdown = """## **RAG Simulator**

RAG 검색 성능을 직접 확인하기 위한 RAG Simulator.
질문을 입력해 어떤 문서들이 검색되는지 확인하고 그 문서를 입력 했을 때 변하는 생성문장을 확인"""

    return markdown
