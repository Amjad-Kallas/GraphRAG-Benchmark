from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

class HaystackVanillaRAG:
    def __init__(
        self,
        embed_model_path,
        llm_generator,
        retrieval_top_k=5,
    ):
        self.document_store = InMemoryDocumentStore()
        self.doc_embedder = SentenceTransformersDocumentEmbedder(
            model=embed_model_path
        )
        self.text_embedder = SentenceTransformersTextEmbedder(
            model=embed_model_path
        )
        self.retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store,
            top_k=retrieval_top_k,
        )
        self.llm = llm_generator

    def index(self, docs):
        documents = [Document(content=d) for d in docs]
        self.doc_embedder.warm_up()
        embedded = self.doc_embedder.run(documents=documents)["documents"]
        self.document_store.write_documents(embedded)

    def rag_qa(self, queries):
        results = []
        for q in queries:
            q_emb = self.text_embedder.run(text=q)["embedding"]
            retrieved = self.retriever.run(query_embedding=q_emb)["documents"]

            context = "\n".join(d.content for d in retrieved)

            prompt = (
                "Answer the question using only the context below.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {q}\nAnswer:"
            )

            answer = self.llm.generate(prompt)

            results.append({
                "question": q,
                "docs": context,
                "answer": answer,
            })
        return results
