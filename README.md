# Lightweight RAG with Local KG, Entity-Aware Retrieval, and Hallucination Detection

This project implements an end-to-end Retrieval-Augmented Generation (RAG) system with the following components:

- Local Knowledge Graph using spaCy and NetworkX
- Entity-Aware Retrieval using Sentence Transformers and FAISS
- Hallucination Detection using NLI with RoBERTa
- FastAPI for serving the API

## Architecture

Wikipedia Slice → Chunking → NER + Entity Linking → Local KG → Dense Embeddings → FAISS → Entity-Aware Retriever → Fusion → Flan-T5-Small → Hallucination Detector → FastAPI

## Setup

1. Build the Docker image:
   ```bash
   docker build -t lightweight-rag .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 lightweight-rag
   ```

3. Send a POST request to `/ask` with a JSON body containing `question`.

## API Usage

```bash
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{"question": "Who is Albert Einstein?"}'
```

## Components

- `rag/kg.py`: Local Knowledge Graph
- `rag/retriever.py`: Entity-Aware Retriever
- `rag/generator.py`: Answer Generator using Flan-T5
- `rag/hallucination.py`: Hallucination Detector
- `rag/fusion.py`: Evidence Fusion
- `api/main.py`: FastAPI application

## Sample Output:
<img width="1402" height="914" alt="image" src="https://github.com/user-attachments/assets/5515ee6c-4fce-4d54-8b73-ac26669b07da" />
