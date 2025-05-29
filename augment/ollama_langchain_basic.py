#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基本的 LangChain 與 Ollama 整合範例
"""

from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def main():
    # 初始化 Ollama LLM
    # 使用 llama3.2:3b 模型，您也可以換成其他已安裝的模型
    llm = Ollama(model="llama3.2:3b")
    
    # 創建一個簡單的提示模板
    prompt = PromptTemplate(
        input_variables=["question"],
        template="請用繁體中文回答以下問題: {question}"
    )
    
    # 創建 LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # 執行鏈並獲取回應
    response = chain.invoke({"question": "什麼是人工智能？"})
    
    # 輸出回應
    print("\n=== 模型回應 ===")
    print(response["text"])
    print("================\n")

if __name__ == "__main__":
    print("開始執行 LangChain 與 Ollama 整合範例...")
    main()
    print("範例執行完畢！")
