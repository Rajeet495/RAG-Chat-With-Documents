\# 📚 Chat With Your Documents — RAG Application



A Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF and ask questions about its content.



\## 🚀 Features



\* Upload PDF documents

\* Extract text from PDFs

\* Split documents into chunks

\* Generate semantic embeddings

\* Store embeddings using FAISS

\* Retrieve relevant document chunks

\* Generate answers using Groq LLM

\* Display retrieved context for transparency

\* Prevent unsupported answers by instructing the model to use only document context



\## 🏗️ RAG Architecture



```text

PDF Document

&#x20;    ↓

Text Extraction

&#x20;    ↓

Text Chunking

&#x20;    ↓

Sentence Embeddings

&#x20;    ↓

FAISS Vector Database

&#x20;    ↓

Semantic Retrieval

&#x20;    ↓

Relevant Context

&#x20;    ↓

Groq LLM

&#x20;    ↓

Answer

```



\## 🛠️ Technologies Used



\* Python

\* Streamlit

\* Groq API

\* Sentence Transformers

\* FAISS

\* PyPDF

\* NumPy



\## 🤖 Embedding Model



`all-MiniLM-L6-v2`



The embedding model converts document chunks and user questions into numerical vector representations for semantic similarity search.



\## 🔍 How It Works



1\. The user uploads a PDF.

2\. Text is extracted from the document.

3\. The extracted text is divided into overlapping chunks.

4\. Each chunk is converted into an embedding.

5\. Embeddings are stored in a FAISS index.

6\. The user's question is also converted into an embedding.

7\. FAISS retrieves the most relevant chunks.

8\. Retrieved chunks are passed as context to the Groq language model.

9\. The model generates an answer based only on the retrieved document context.



\## ▶️ Run Locally



Install dependencies:



```bash

pip install -r requirements.txt

```



Configure the Groq API key in:



```text

.streamlit/secrets.toml

```



Example:



```toml

GROQ\_API\_KEY = "your\_api\_key\_here"

```



Run the application:



```bash

python -m streamlit run app.py

```



\## ⚠️ Security



Never upload `.streamlit/secrets.toml` or expose your Groq API key publicly.



The API key is excluded from Git using `.gitignore`.



\## 📌 Project Purpose



This project was developed as part of a Generative AI internship module covering:



\* Retrieval-Augmented Generation

\* Embeddings

\* Semantic similarity

\* Document processing

\* Text chunking

\* Vector databases

\* Information retrieval

\* LLM-based question answering



\## 👨‍💻 Author



\*\*Rajeet Pratap Mall\*\*



B.Tech CSE — Artificial Intelligence \& Machine Learning



