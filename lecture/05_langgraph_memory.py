import os
from dotenv import load_dotenv, find_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import MemorySaver

class GenLLM:
    def __init__(self):
        _ = load_dotenv(find_dotenv())
        
        self.llm = ChatOpenAI(
            api_key = os.getenv("OPENAI_API_KEY"),
            model = "gpt-4o-mini",
            temperature = 0,
        )
        
        # State 정의
        self.workflow = StateGraph(state_schema=MessagesState)
        
        def call_model(state: MessagesState):
            response = self.llm.invoke(state["messages"])
            return {"messages": response}
        
        self.workflow.add_node("model", call_model)
        self.workflow.add_edge(START, "model")
        
        memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=memory)
        
    def run(self, thread_id: str, user_input: str) -> str:
        human_msg = HumanMessage(content=user_input)
        config = {"configurable": {"thread_id": thread_id}}
        result = self.app.invoke({"messages": [human_msg]}, config=config)
        ai_msg = result["messages"][-1]
        print(f"Assistant: {ai_msg.content}")
        return ai_msg.content        
    
gen_llm = GenLLM()
thread = "user123"
while True:
    q = input("Human: ").strip()
    if not q:
        continue
    gen_llm.run(thread, q)