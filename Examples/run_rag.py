import os
import asyncio
import argparse
import json
import logging
from typing import Dict, List
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
import requests

# ============================================================
# Setup
# ============================================================

load_dotenv()
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("rag_processing.log"),
    ],
)

# ============================================================
# Utility functions (copied from HippoRAG script)
# ============================================================

def group_questions_by_source(question_list: List[dict]) -> Dict[str, List[dict]]:
    grouped = {}
    for q in question_list:
        src = q["source"]
        grouped.setdefault(src, []).append(q)
    return grouped


def split_text(
    text: str,
    tokenizer: AutoTokenizer,
    chunk_token_size: int = 256,
    chunk_overlap_token_size: int = 32,
) -> List[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_token_size, len(tokens))
        chunk = tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        chunks.append(chunk)
        if end == len(tokens):
            break
        start += chunk_token_size - chunk_overlap_token_size
    return chunks


# ============================================================
# vLLM Generator (OpenAI-compatible)
# ============================================================

class VLLMGenerator:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self.url = f"{base_url}/v1/chat/completions"
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        r = requests.post(self.url, json=payload, timeout=300)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# ============================================================
# Vanilla RAG (Haystack-based)
# ============================================================

class VanillaRAG:
    def __init__(
        self,
        embed_model_path: str,
        llm: VLLMGenerator,
        retrieval_top_k: int = 5,
    ):
        self.document_store = InMemoryDocumentStore()
        self.doc_embedder = SentenceTransformersDocumentEmbedder(
            model=embed_model_path
        )
        self.text_embedder = SentenceTransformersTextEmbedder(
            model=embed_model_path
        )

        self.doc_embedder.warm_up()
        self.text_embedder.warm_up()

        self.retriever = InMemoryEmbeddingRetriever(
            document_store=self.document_store,
            top_k=retrieval_top_k,
        )
        self.llm = llm

    def index(self, docs: List[str]):
        documents = [Document(content=d) for d in docs]
        
        embedded = self.doc_embedder.run(documents=documents)["documents"]
        self.document_store.write_documents(embedded)

    def rag_qa(self, queries: List[str]):
        outputs = []
        for q in queries:
            q_emb = self.text_embedder.run(text=q)["embedding"]
            retrieved = self.retriever.run(query_embedding=q_emb)["documents"]

            # ✅ list of chunk strings (what you want in JSON)
            context_chunks = [doc.content for doc in retrieved]

            # build prompt AS CLOSE AS possible to the original pipeline
            prompt_user = ''
            for passage in context_chunks:
                prompt_user += f'Wikipedia Title: {passage}\n\n'

            prompt_user += f'Question: {q}\nAnswer: '

            prompt = (
                'As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. '
                'Your response start after "Answer: " to present a concise, definitive response, devoid of additional elaborations.'
                '\n\n'
                + prompt_user
            )
            answer = self.llm.generate(prompt)

            outputs.append({
                "question": q,
                "docs": context_chunks,   
                "answer": answer,
            })
        return outputs


# ============================================================
# Corpus processing (mirrors HippoRAG2)
# ============================================================

def process_corpus(
    corpus_name: str,
    context: str,
    base_dir: str,
    embed_model_path: str,
    llm_base_url: str,
    llm_model_name: str,
    questions: Dict[str, List[dict]],
    sample: int,
    rag,
):
    logging.info(f"📚 Processing corpus: {corpus_name}")

    output_dir = f"./results/rag/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")

    tokenizer = AutoTokenizer.from_pretrained(embed_model_path)
    chunks = split_text(context, tokenizer)
    docs = [f"{i}:{chunk}" for i, chunk in enumerate(chunks)]

    # docs = docs[:10]

    corpus_questions = questions.get(corpus_name, [])
    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]

    all_queries = [q["question"] for q in corpus_questions]

    llm = VLLMGenerator(
        base_url=llm_base_url,
        model=llm_model_name,
    )



    rag.index(docs)
    solutions = rag.rag_qa(all_queries)

    results = []
    for q in corpus_questions:
        sol = next(s for s in solutions if s["question"] == q["question"])
        results.append({
            "id": q["id"],
            "question": q["question"],
            "source": corpus_name,
            "context": sol["docs"],
            "evidence": q.get("evidence", ""),
            "question_type": q.get("question_type", ""),
            "generated_answer": sol["answer"],
            "ground_truth": q["answer"],
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"💾 Saved {len(results)} predictions to {output_path}")



def main():
    SUBSET_PATHS = {
        "medical": {
            "corpus": "./Datasets/Corpus/medical.parquet",
            "questions": "./Datasets/Questions/medical_questions.parquet",
        },
        "novel": {
            "corpus": "./Datasets/Corpus/novel.parquet",
            "questions": "./Datasets/Questions/novel_questions.parquet",
        },
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True, choices=["medical", "novel"])
    parser.add_argument("--base_dir", default="./rag_workspace")
    parser.add_argument("--embed_model_path", required=True)
    parser.add_argument("--llm_base_url", default="http://localhost:8000")
    parser.add_argument("--llm_model_name", required=True)
    parser.add_argument("--sample", type=int, default=None)

    args = parser.parse_args()

    # --------------------------------------------------
    # Load corpus
    # --------------------------------------------------
    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    corpus_ds = load_dataset("parquet", data_files=corpus_path, split="train")
    corpus_data = [
        {"corpus_name": x["corpus_name"], "context": x["context"]}
        for x in corpus_ds
    ]

    if args.sample:
        corpus_data = corpus_data[:1]

    # --------------------------------------------------
    # Load questions
    # --------------------------------------------------
    questions_path = SUBSET_PATHS[args.subset]["questions"]
    questions_ds = load_dataset("parquet", data_files=questions_path, split="train")
    questions = [{
        "id": q["id"],
        "source": q["source"],
        "question": q["question"],
        "answer": q["answer"],
        "question_type": q["question_type"],
        "evidence": q["evidence"],
    } for q in questions_ds]

    grouped_questions = group_questions_by_source(questions)



    # --------------------------
    # Sequential Version
    # --------------------------
    llm = VLLMGenerator(
        base_url=args.llm_base_url,
        model=args.llm_model_name,
    )

    for item in corpus_data:
        rag = VanillaRAG(
            embed_model_path=args.embed_model_path,
            llm=llm,
            retrieval_top_k=5,
        )

        process_corpus(
            corpus_name=item["corpus_name"],
            context=item["context"],
            base_dir=args.base_dir,
            embed_model_path=args.embed_model_path,
            llm_base_url=args.llm_base_url,
            llm_model_name=args.llm_model_name,
            questions=grouped_questions,
            sample=args.sample,
            rag=rag,
        )

    # --------------------------
    # Async version (optional)
    # --------------------------
    '''
    async def run_all():
        tasks = []
        for item in corpus_data:
            tasks.append(asyncio.to_thread(
                process_corpus,
                item["corpus_name"],
                item["context"],
                args.base_dir,
                args.embed_model_path,
                args.llm_base_url,
                args.llm_model_name,
                grouped_questions,
                args.sample,
            ))
        await asyncio.gather(*tasks)

    asyncio.run(run_all())'''


if __name__ == "__main__":
    main()
