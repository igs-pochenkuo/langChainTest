#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用 LangChain 與本地 Ollama ChatOllama 模型互動的範例
"""

from langchain_ollama import ChatOllama

# 建立訊息格式，支援多輪對話
# 每個訊息為 (角色, 內容)，角色可為 system, human, assistant

def main():
    # 初始化 ChatOllama，指定模型名稱
    llm = ChatOllama(
        model="llama3.2:3b"
    )

    # 單輪對話
    print("\n=== 簡單對話測試 ===")
    messages = [
        ("human", "請用繁體中文介紹一下你自己")
    ]
    response = llm.invoke(messages)
    print("AI: ", response.content)

    # 多輪對話
    print("\n=== 多輪對話測試 ===")
    messages = [
        ("human", "你好，請用繁體中文回答"),
        ("human", "請告訴我一個關於人工智慧的笑話")
    ]
    response = llm.invoke(messages)
    print("AI: ", response.content)

if __name__ == "__main__":
    main() 