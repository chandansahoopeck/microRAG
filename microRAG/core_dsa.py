import numpy as np

class MiniVectorStore:
    def __init__(self):
        # store the RAW chunks
        self.documents = []
        # Store the 384-dimensional embeddings vectors
        self.embeddings = []

    def add_document(self, chunks: list[str], chunks_embeddings: list[list[float]]):
        """Indexes new documents into our local memory"""
        self.documents.extend(chunks)
        self.embeddings.extend(chunks_embeddings)

    def similarity_search(self, query_embedding: list[float], top_k: int = 3) -> list[str]:
        if not self.embeddings:
            return []
        
        q_vec = np.array(query_embedding)
        doc_vecs = np.array(self.embeddings)

        dot_products = np.dot(doc_vecs, q_vec)
        q_mag = np.linalg.norm(q_vec)
        doc_mag = np.linalg.norm(doc_vecs, axis=1)

        similarities = dot_products / (doc_mag * q_mag)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]