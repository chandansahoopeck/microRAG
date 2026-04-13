import json
from sentence_transformers import SentenceTransformer
# Recommended to update this import eventually!
from langchain_community.llms import Ollama
from core_dsa import MiniVectorStore

class RAGPipeline:
    def __init__(self):
        self.embedder = SentenceTransformer('./local-all-MiniLM-L6-v2')
        self.vector_store = MiniVectorStore()
        self.llm = Ollama(model='llama3', temperature=0.0, format="json")

    def ingest_txt(self, text: str):
        chunks = [chunk.strip() for chunk in text.split('\n\n') if len(chunk.strip()) > 50]
        embeddings = self.embedder.encode(chunks).tolist()
        
        # NOTE: Make sure 'add' is actually the correct method name in MiniVectorStore!
        self.vector_store.add_document(chunks, embeddings) 
        
        return len(chunks)

    def process_claim(self, user_query: str) -> dict:
        # Fixed: passed string directly instead of a list to get a flat vector
        query_vec = self.embedder.encode(user_query).tolist()
        
        # Fixed: variable name and spelling
        relevant_chunks = self.vector_store.similarity_search(query_vec, top_k=2)
        
        context = "\n---\n".join(relevant_chunks)

        prompt = f"""
        you are an Intelligent claim processing Automation triage agent.
        CONTEXT (Past similarity Claim rules):
        {context}

        NEW CLAIM to CLASSIFY:
        {user_query}

        Respond only in JSON format:
        {{"category": "Fraud|Standard|Priority", "reasoning": "Why"}}
        """

        # Fixed: using .invoke() instead of calling the object directly
        # Ollama's response object usually has a .content attribute holding the string
        response_obj = self.llm.invoke(prompt)
        
        # Depending on the exact LangChain version, it might return a string or an AIMessage
        response_text = response_obj if isinstance(response_obj, str) else response_obj.content
        
        return json.loads(response_text)