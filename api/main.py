from fastapi import FastAPI
from rag.retriever import EntityAwareRetriever
from rag.generator import Generator
from rag.hallucination import HallucinationDetector
from rag.fusion import evidence_fusion

# Dummy chunks for demonstration
chunks = [
    "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.",
    "The theory of relativity includes special relativity and general relativity.",
    "Einstein won the Nobel Prize in Physics in 1921 for his discovery of the law of the photoelectric effect.",
    "Berlin is the capital and largest city of Germany by both area and population.",
    "Germany is a country in Central Europe.",
    "The photoelectric effect is the emission of electrons when electromagnetic radiation hits a metal surface."
]

app = FastAPI()

retriever = EntityAwareRetriever(chunks)
generator = Generator()
hallucination = HallucinationDetector()

@app.post("/ask")
def ask(question: str):
    docs = retriever.retrieve(question, k=5)
    context = evidence_fusion(docs)
    answer = generator.generate(context, question)

    hallu = hallucination.check(answer, docs)

    return {
        "question": question,
        "answer": answer,
        "hallucination": hallu,
        "evidence": docs
    }