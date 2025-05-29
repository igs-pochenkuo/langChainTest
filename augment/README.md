# LangChain 與 Ollama 整合範例

這個專案包含了幾個使用 LangChain 與本地 Ollama 模型整合的範例。

## 前提條件

1. 已安裝 Python 3.9+
2. 已安裝 Ollama (https://ollama.ai/)
3. 已下載至少一個 Ollama 模型 (例如 llama3.2:3b)

## 安裝依賴

```bash
pip install langchain langchain-community langchain-ollama
```

## 範例文件

本專案包含三個範例：

### 1. 基本範例 (ollama_langchain_basic.py)

展示了如何使用 LangChain 連接 Ollama 模型並進行簡單的問答。

```bash
python ollama_langchain_basic.py
```

### 2. 進階範例 (ollama_langchain_advanced.py)

展示了如何使用 LangChain 的對話記憶功能，實現多輪對話。

```bash
python ollama_langchain_advanced.py
```

### 3. RAG 範例 (ollama_langchain_rag.py)

展示了如何使用 LangChain 的檢索增強生成 (RAG) 功能，從文檔中檢索相關信息並生成回答。

```bash
python ollama_langchain_rag.py
```

## 自定義模型

您可以修改範例中的 `model` 參數，使用您已安裝的任何 Ollama 模型。例如：

```python
llm = Ollama(model="llama3.2:1b")  # 使用較小的模型
```

或

```python
llm = Ollama(model="nemotron-mini")  # 使用其他模型
```

使用 `ollama list` 命令查看您已安裝的模型列表。

## 注意事項

- 首次運行時，Ollama 可能需要一些時間來加載模型。
- RAG 範例會在當前目錄創建一個 `chroma_db` 文件夾來存儲向量數據庫。
- 如果遇到性能問題，可以嘗試使用較小的模型，如 `llama3.2:1b`。
