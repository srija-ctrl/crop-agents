"""Central orchestrator: routes the task, delegates to agents, keeps the shared state,
and allows AT MOST ONE review-triggered revision (steps 6-7)."""
from __future__ import annotations
from typing import Any, Dict

from .agents import (CropRecommendationAgent, KnowledgeAgent, RouterAgent, SoilAnalysisAgent,
                     ValidationAgent, WriterAgent)
from .config import MAX_REVISIONS, MODEL_PATH
from .data import DatasetStats, load_dataset
from .llm import OllamaLLM
from .model_tool import CropPredictorTool, train_and_save
from .rag import KnowledgeBase
from .report import build_report
from .state import log, new_state


class Orchestrator:
    def __init__(self, llm: OllamaLLM, predictor: CropPredictorTool, stats: DatasetStats, kb: KnowledgeBase):
        self.llm, self.predictor, self.stats, self.kb = llm, predictor, stats, kb
        self.router = RouterAgent(llm, stats)
        self.soil = SoilAnalysisAgent(stats)
        self.crop = CropRecommendationAgent(predictor, stats)
        self.knowledge = KnowledgeAgent(kb)
        self.writer = WriterAgent(llm)
        self.reviewer = ValidationAgent(llm, stats)

    def run(self, user_input: Dict[str, Any], demo_revision: bool = False) -> Dict[str, Any]:
        state = new_state(user_input, demo_revision)
        log(state, "Orchestrator", "received task")

        # 1) route (Ollama classifies intent; rules validate the numbers)
        self.router.run(state)
        if state["route"] in ("invalid", "unsupported"):
            state["review_status"] = "not_needed"
            return self._finish(state)

        # 2) delegate to the specialised agents
        if state["route"] == "recommend":
            self.soil.run(state)
            self.crop.run(state)          # prediction comes from the dataset-derived tool
        self.knowledge.run(state)         # RAG over the text corpus

        # 3) unsupported evidence -> safe response, no invented advice
        if not state["retrieved"]:
            state["review_status"] = "skipped_no_evidence"
            log(state, "Orchestrator", "no evidence retrieved -> advisory withheld")
            return self._finish(state)

        # 4) draft -> review -> (at most one) revision
        self.writer.draft(state)
        self.reviewer.review(state)
        if state["review"]["verdict"] == "revise" and state["retry_count"] < MAX_REVISIONS:
            state["retry_count"] += 1
            log(state, "Orchestrator", f"review requested changes -> revision #{state['retry_count']}")
            self.writer.revise(state)
            self.reviewer.review(state)
            state["review_status"] = ("approved_after_revision" if state["review"]["verdict"] == "approve"
                                      else "accepted_with_warnings")
        else:
            state["review_status"] = "approved" if state["review"]["verdict"] == "approve" else "accepted_with_warnings"
        log(state, "Orchestrator", f"final review status: {state['review_status']}")
        return self._finish(state)

    def _finish(self, state: Dict[str, Any]) -> Dict[str, Any]:
        acc = self.predictor.metrics.get("accuracy")
        state["final_response"] = build_report(state, acc)
        state["report_text"] = state["final_response"]["text"]
        return state


def build_system(force_offline: bool = False, rebuild_kb: bool = False, force_tfidf: bool = False,
                 verbose: bool = True) -> Orchestrator:
    """Load data, model (trains it if missing), Ollama client and knowledge base."""
    df = load_dataset()
    stats = DatasetStats(df)
    if not MODEL_PATH.exists():
        if verbose:
            print("[setup] No trained model found - training one now (run train.py to see the full report)...")
        train_and_save(df)
    predictor = CropPredictorTool.load()
    llm = OllamaLLM(force_offline=force_offline)
    kb = KnowledgeBase(llm, force_tfidf=force_tfidf or force_offline)
    backend = kb.build(rebuild=rebuild_kb)
    if verbose:
        chat = "yes" if llm.chat_available() else "NO"
        print(f"[setup] rows={len(df)} | Ollama chat model '{llm.model}' available: {chat} | RAG backend: {backend}")
        if llm.status_message:
            print(f"[setup] note: {llm.status_message}")
    return Orchestrator(llm, predictor, stats, kb)
