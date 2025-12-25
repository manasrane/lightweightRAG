from transformers import pipeline

class Generator:
    def __init__(self):
        self.pipe = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device=-1
        )

    def generate(self, context, question):
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        result = self.pipe(prompt, max_length=200, do_sample=False)
        return result[0]['generated_text']