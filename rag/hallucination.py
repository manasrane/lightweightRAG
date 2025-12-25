from transformers import pipeline

class HallucinationDetector:
    def __init__(self):
        self.nli = pipeline(
            "text-classification",
            model="roberta-large-mnli",
            device=-1
        )

    def check(self, answer, evidence):
        verdicts = []
        for ev in evidence:
            inp = f"{ev} </s></s> {answer}"
            result = self.nli(inp)[0]
            verdicts.append(result["label"] == "ENTAILMENT")

        score = sum(verdicts) / len(verdicts)
        return {
            "hallucinated": score < 0.5,
            "support_score": score
        }