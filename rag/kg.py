import spacy
import networkx as nx
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

class LocalKG:
    def __init__(self):
        self.graph = nx.Graph()

    def extract_entities(self, text):
        doc = nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "DATE"}]

    def add_chunk(self, chunk):
        entities = self.extract_entities(chunk)
        for e in entities:
            self.graph.add_node(e)
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                self.graph.add_edge(entities[i], entities[j])

    def build(self, chunks):
        for c in chunks:
            self.add_chunk(c)