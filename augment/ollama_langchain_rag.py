#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LangChain 與 Ollama 整合的 RAG (檢索增強生成) 範例
"""

import os
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader

def create_sample_document():
    """創建一個示例文檔用於演示"""
    sample_text = """
    人工智能（AI）簡介
    
    人工智能是計算機科學的一個分支，致力於創建能夠模擬人類智能的系統。
    
    機器學習是人工智能的一個子領域，它使用統計技術使計算機系統能夠從數據中"學習"，
    而無需明確編程。深度學習是機器學習的一個子集，使用神經網絡進行學習。
    
    自然語言處理（NLP）是人工智能的另一個重要領域，專注於計算機理解、解釋和生成人類語言。
    
    計算機視覺是人工智能的一個領域，專注於使計算機能夠從數字圖像或視頻中獲取高級理解。
    
    強化學習是一種機器學習方法，其中智能體通過與環境互動並從其行為的結果中學習。
    
    生成式AI是人工智能的一個分支，專注於創建新內容，如文本、圖像、音樂或其他媒體。
    
    人工智能的應用包括：
    - 虛擬助手（如Siri、Alexa）
    - 推薦系統（如Netflix、YouTube）
    - 自動駕駛車輛
    - 醫療診斷
    - 金融交易
    - 遊戲（如AlphaGo）
    
    人工智能面臨的挑戰包括：
    - 數據隱私和安全
    - 算法偏見
    - 透明度和可解釋性
    - 就業影響
    - 倫理考量
    """
    
    with open("ai_intro.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)
    
    return "ai_intro.txt"

def main():
    # 創建示例文檔
    document_path = create_sample_document()
    print(f"已創建示例文檔: {document_path}")
    
    # 加載文檔
    loader = TextLoader(document_path, encoding="utf-8")
    documents = loader.load()
    
    # 分割文本
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n"
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"文檔已分割為 {len(chunks)} 個區塊")
    
    # 使用 Ollama 創建嵌入
    embeddings = OllamaEmbeddings(model="llama3.2:3b")
    
    # 創建向量存儲
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    # 創建檢索器
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # 初始化 Ollama LLM
    llm = Ollama(model="llama3.2:3b")
    
    # 創建 QA 鏈
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )
    
    # 提問並獲取回答
    questions = [
        "什麼是人工智能？",
        "機器學習是什麼？",
        "人工智能有哪些應用？",
        "人工智能面臨哪些挑戰？"
    ]
    
    for question in questions:
        print(f"\n問題: {question}")
        result = qa_chain.invoke({"query": question})
        print(f"回答: {result['result']}")
        print("-" * 50)
    
    # 清理
    if os.path.exists("./chroma_db"):
        print("注意: 向量數據庫保留在 ./chroma_db 目錄中")

if __name__ == "__main__":
    print("開始執行 LangChain 與 Ollama 的 RAG 範例...")
    main()
    print("範例執行完畢！")
