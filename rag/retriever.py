from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

class EntityAwareRetriever:
    def __init__(self, chunks):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chunks = chunks
        self.emb = self.model.encode(chunks)
        self.index = faiss.IndexFlatL2(self.emb.shape[1])
        self.index.add(self.emb)

    def extract_entities(self, text):
        doc = nlp(text)
        return {ent.text for ent in doc.ents}

    def retrieve(self, query, k=5):
        q_emb = self.model.encode([query])
        _, idx = self.index.search(q_emb, k * 2)

        query_entities = self.extract_entities(query)
        scored = []

        for i in idx[0]:
            chunk = self.chunks[i]
            chunk_entities = self.extract_entities(chunk)
            entity_overlap = len(query_entities & chunk_entities)
            scored.append((entity_overlap, chunk))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [c for _, c in scored[:k]]