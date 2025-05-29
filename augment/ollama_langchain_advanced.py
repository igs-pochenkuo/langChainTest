#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
進階的 LangChain 與 Ollama 整合範例
包含對話記憶和鏈式調用
"""

from langchain_community.llms import Ollama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

def main():
    # 初始化 Ollama LLM
    # 您可以選擇您已安裝的任何模型
    model_name = "llama3.2:3b"  # 可以換成 "llama3.2:1b" 或 "nemotron-mini"
    
    llm = Ollama(model=model_name)
    
    # 創建對話記憶
    memory = ConversationBufferMemory()
    
    # 創建對話提示模板
    template = """
    以下是人類和AI助手之間的友好對話。
    AI助手樂於提供幫助、創造性和友好。
    AI助手應該用繁體中文回答。

    當前對話歷史：
    {history}
    
    人類: {input}
    AI助手:"""
    
    prompt = PromptTemplate(
        input_variables=["history", "input"],
        template=template
    )
    
    # 創建對話鏈
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True  # 設置為True以顯示鏈的執行過程
    )
    
    # 進行多輪對話
    questions = [
        "你好，請介紹一下自己",
        "什麼是機器學習？",
        "你能給我一個Python的例子嗎？"
    ]
    
    for question in questions:
        print(f"\n人類: {question}")
        response = conversation.invoke({"input": question})
        print(f"AI助手: {response['response']}")
        print("-" * 50)
    
    # 顯示記憶中存儲的對話歷史
    print("\n=== 對話歷史 ===")
    print(memory.buffer)
    print("================\n")

if __name__ == "__main__":
    print("開始執行進階 LangChain 與 Ollama 整合範例...")
    main()
    print("範例執行完畢！")
