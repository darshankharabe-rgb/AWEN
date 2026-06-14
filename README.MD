# AWEN: Enterprise RAG Engine with Intelligent Caching

AWEN is a high-performance, full-stack Retrieval-Augmented Generation (RAG) platform designed to parse, index, and query complex documents in real time. Built with a decoupled microservices architecture, the system utilizes a high-speed vector database for semantic search, an in-memory database for query caching, and complete containerization for seamless local or cloud deployment.

## 🚀 Key Architectural Highlights

* **Hybrid RAG Pipeline:** Efficiently ingests PDFs, processes text chunks, maps them to vector spaces using the `all-MiniLM-L6-v2` embedding model, and stores them in **ChromaDB** for highly accurate semantic retrieval.
* **Performance Optimization (Redis Caching):** Features an intelligent caching layer powered by **Redis**. Identical and high-frequency queries bypass the vector database and LLM pipeline entirely, resulting in sub-millisecond response times (`⚡ REDIS CACHE HIT`) and significantly minimized API computation costs.
* **Decoupled & Scalable Architecture:** Built as two separate microservices (FastAPI backend and React frontend) interacting via a secure, authenticated RESTful API layer.
* **Production-Ready Containerization:** Fully orchestrated using **Docker** and **Docker Compose**, ensuring identical runtime environments across local development and cloud production systems.

---

## 🛠️ Tech Stack

| Layer | Technologies Used |
| :--- | :--- |
| **Frontend** | React, Vite, Tailwind CSS |
| **Backend API** | FastAPI, Python, Pydantic, Uvicorn |
| **Vector Database & AI** | ChromaDB, LangChain/LlamaIndex, `all-MiniLM-L6-v2` Embeddings, Gemini API |
| **Caching Layer** | Redis Engine |
| **DevOps & Orchestration** | Docker, Docker Compose |

---

## 🏗️ System Architecture Flow

```mermaid
graph TD
    A[React Frontend] -->|1. HTTP Request| B[FastAPI Gateway]
    B -->|2. Check Cache| C{Redis Cache}
    C -->|Cache Hit: Return Fast| A
    C -->|Cache Miss: Route Query| D[ChromaDB Vector Store]
    D -->|3. Extract Relevant Text Chunks| E[LLM Engine / Gemini]
    E -->|4. Generate Contextual Response| B
    B -->|5. Update Cache & Return Answer| C
