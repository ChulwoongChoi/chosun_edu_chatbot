# 기존 챗봇 서비스들은
# 1. LangChain
#  프롬프트 -> LLM -> 결과 : 심플, 단뱡항 구조

# 2. LangGrpah
#  - Graph를 구성하는 Node -> LangChain
#  프롬프트 -> LLM -> 프롬프트 분석 -> LLM -> 프롬프트 확장 -> LLM -> 결과 -> LLM -> 평가


from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
import os
from langchain.prompts import ChatPromptTemplate


# 1. env 불러오기
_ = load_dotenv(find_dotenv())

# 2. LLM 모델 생성
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)

# 3. 프롬프트 생성
system_prompt = """
    You are a helpful assistant.
    You will be provided with a set of documents and a question.
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. You will be provided with a set of documents and a question."),
    ("human", "나의 이름은 최철웅이야!")
])

# 4. Chain 생성
#  -> 방향으로만 실행!
chain = prompt | llm

# 5. Chain 실행
question = "나의 이름은 최철웅이야!"
result = chain.invoke({"question": question})
print(result.content)