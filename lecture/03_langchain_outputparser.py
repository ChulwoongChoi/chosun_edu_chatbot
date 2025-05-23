import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pprint import pprint

_ = load_dotenv(find_dotenv())

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
)

systme_prompt = """
    너는 위키 백과사전의 편집자야.
    작가는 소설에 사용할 "키워드"의 배경과 내용들의 정보가 필요합니다.
    전달받은 키워드를 다음 조건에 맞게 출력하세요.
    
    wiki_keyowrd: {keyword}
    format_instructions: {format_instructions}
    
    조건1. 한글로 작성하세요.
    조건2. 위키백과에서 많이 사용하는 문체와 어조를 사용하세요.
    조건3. 새로운 글을 생성하지 마세요.
    조건4. 출력되는 형태는 format_instructions 출력 형식의 빈값을 채워서 출력해주세요.
           JSON Key값 중 자료가 없는 항목은 빈칸으로 두세요.
"""

format_instructions = """
    JSON 출력 형식:
    output_json = {
        "input_keyword": ,
        "related_entries": {
            "Concept_Definition": {
                "description": ""
            },
            "role_in_worldview": {
                "description": ""
            },
            "Origin and History": {
                "name": "",
                "description": "",
            }
            "main_characters": [
                {
                    "name": "",
                    "description": "",
                }, 
                {
                    "name": "",
                    "description": "",
                }, 
            ],
            "key_events_in_story": [
                "",
                "",
                ""
            ],
            "related_items": [
                {
                    "name": "",
                    "description": ""
                },
                {
                    "name": "",
                    "description": ""
                }
            ],
            "Similar concepts or terms": {
                    "name": "",
                    "description": ""
            },
            "representation_in_webtoons": {
                "general_description": "",
                "visual_effects": "",
                "notable_webtoons": [
                    "",
                    ""
                ]
            }
        },
    }
"""

prompt = PromptTemplate(
    template=systme_prompt,
    input_variables=["keyword"],
    partial_variables={"format_instructions": format_instructions}
)

chain = prompt | llm | JsonOutputParser()
result = chain.invoke({"keyword": "드래곤"})

print(result)