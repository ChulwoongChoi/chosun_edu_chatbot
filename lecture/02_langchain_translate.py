import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

_ = load_dotenv(find_dotenv())

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)

system_prompt = """
    너는 소설을 쓰는 작가를 보조하는 역할이에요.
    전달받은 글을 다음 조건에 맞게 한국어에서 '{lang}'로 번역해주세요.
    
    조건1. 작가의 질문에 절대로 답변하지마세요.
    조건2. 작가의 글을 문자열 그대로 번역해주세요.
    조건3. 문법에 신경써서 번역해주세요.
    조건4. 줄임말과 신조어 등은 최대한 사용하지마세요.
    조건5. 판타지 소설에서 많이 사용하는 문체와 어조로 번역해주세요.
    조건6. 번역시 기존 글의 스타일을 유지해주세요.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{novel}")
])

novel = """
    어둠의 숲 가장자리에 선 소녀 리아는
    달빛에 은빛으로 빛나는 고양이 눈을 한 용을 처음 마주했다.
    용의 어깨 위엔 고대 룬 문자가 반짝이며,
    그 문자는 천 년 전 봉인된 마왕의 저주를 품고 있었다.
    리아가 손을 내밀자, 용은 낮게 으르렁거리며 대지를 흔들었다.
    하지만 소녀의 순수한 마음이 용의 어둠을 깨우고,
    붉은 불꽃 대신 따뜻한 빛이 용의 숨결에 실려 퍼졌다.
    두 존재는 운명처럼 손을 잡고, 숲 속 고대 성으로 발걸음을 옮겼다.
    그곳에선 잊힌 왕좌와 잠들어 있던 마왕이 깨어날 준비를 하고 있었다.
    리아와 용은 희망의 불씨를 손에 쥐고, 운명의 결전을 향해 나아갔다.
"""
lang = "영어"

novel_input = {
    "novel": novel,
    "lang": lang
}

chain = prompt | llm
result = chain.invoke(novel_input)
print(result.content)