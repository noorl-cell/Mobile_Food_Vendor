"""
Stage 7: RAG Chatbot Agent
TF-IDF retrieval over rules + permits + violation-category examples.
Generation uses Anthropic API if ANTHROPIC_API_KEY is set, otherwise falls
back to a clean template that cites retrieved sources.
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
RULES = ROOT / "rules"
PROC = ROOT / "data" / "processed"


def build_knowledge_base():
    """Turn rules + permits + category exemplars into a list of passages."""
    passages = []

    rules = json.loads((RULES / "extracted_rules.json").read_text())
    for r in rules:
        body = r.get("requirement_summary", "")
        section = r.get("section", "")
        passages.append({
            "id": f"rule::{r['rule_id']}",
            "title": f"Rule: {r['rule_id']}" + (f" ({section})" if section else ""),
            "text": body,
            "source": r["source"],
            "kind": "regulation",
        })

    permits = json.loads((RULES / "required_permits.json").read_text())
    for p in permits["permits"]:
        text = (
            f"{p['name']}. Issued by {p['issuer']}. "
            f"Required for: {p['required_for']} "
            f"Legal authority: {p['authority']}. "
            f"Typical fee: ${p['typical_fee_usd']}. Renewal: {p['renewal']}. "
            f"Prerequisites: {'; '.join(p['prerequisites'])}."
        )
        passages.append({
            "id": f"permit::{p['id']}",
            "title": p["name"],
            "text": text,
            "source": p["source"],
            "kind": "permit",
        })
    for cp in permits.get("conditional_permits", []):
        passages.append({
            "id": f"permit::{cp['id']}",
            "title": cp["name"],
            "text": f"{cp['name']}: needed when {cp['when']}. Issuer: {cp['issuer']}.",
            "source": cp["source"],
            "kind": "conditional_permit",
        })

    # Add top violation category exemplars from real Suffolk data
    try:
        vdf = pd.read_csv(PROC / "suffolk_violations_clean.csv", usecols=["violation_text"])
        sample = (
            vdf["violation_text"].dropna().astype(str)
            .value_counts().head(20).reset_index()
        )
        sample.columns = ["text", "count"]
        for _, row in sample.iterrows():
            passages.append({
                "id": f"violation_example::{hash(row['text']) & 0xffff}",
                "title": f"Common violation (seen {row['count']}x)",
                "text": row["text"][:500],
                "source": "Suffolk County Restaurant Violations 2024-2025",
                "kind": "violation_example",
            })
    except FileNotFoundError:
        pass

    return passages


class RAG:
    def __init__(self):
        self.passages = build_knowledge_base()
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        corpus = [f"{p['title']}. {p['text']}" for p in self.passages]
        self.matrix = self.vec.fit_transform(corpus)

    def retrieve(self, query: str, k: int = 5):
        q = self.vec.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        idx = np.argsort(-scores)[:k]
        return [
            {**self.passages[i], "score": round(float(scores[i]), 3)}
            for i in idx if scores[i] > 0
        ]

    def answer(self, query: str, k: int = 5):
        hits = self.retrieve(query, k=k)
        if not hits:
            return {
                "answer": "I don't have information on that in my knowledge base. "
                          "Try asking about permits, inspections, or food-safety rules.",
                "sources": [],
            }

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            try:
                answer = self._llm_answer(query, hits, api_key)
                return {"answer": answer, "sources": hits, "mode": "llm"}
            except Exception as e:
                # fall through to template
                pass

        return {"answer": self._template_answer(query, hits), "sources": hits, "mode": "template"}

    def _template_answer(self, query: str, hits):
        lines = [f"Based on **{len(hits)}** relevant sources in the knowledge base:\n"]
        for i, h in enumerate(hits, 1):
            snippet = h["text"][:280].replace("\n", " ")
            lines.append(f"**{i}. {h['title']}** ({h['kind']})")
            lines.append(f"> {snippet}{'...' if len(h['text']) > 280 else ''}")
            lines.append(f"_Source: {h['source']}_\n")
        lines.append(
            "\n*For a conversational answer with an LLM, set the "
            "`ANTHROPIC_API_KEY` environment variable before launching the app.*"
        )
        return "\n".join(lines)

    def _llm_answer(self, query, hits, api_key):
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        context = "\n\n".join(
            f"[{i+1}] {h['title']} ({h['kind']})\n{h['text']}\nSource: {h['source']}"
            for i, h in enumerate(hits)
        )
        sys_prompt = (
            "You are a compliance assistant for mobile food vendors operating in "
            "Suffolk County, NY. Answer the user's question using ONLY the provided "
            "context. Cite sources inline like [1], [2]. If the context does not "
            "contain the answer, say so. Be concise and practical."
        )
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            system=sys_prompt,
            messages=[{
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            }],
        )
        return msg.content[0].text


if __name__ == "__main__":
    rag = RAG()
    print(f"Knowledge base: {len(rag.passages)} passages\n")
    for q in [
        "What permits do I need to start a food truck in Suffolk County?",
        "How cold should I keep cold food?",
        "Do I need a commissary?",
    ]:
        print(f"Q: {q}")
        out = rag.answer(q, k=3)
        print(f"Mode: {out.get('mode')}")
        print(out["answer"][:500])
        print("---")
