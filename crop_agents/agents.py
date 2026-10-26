"""The specialised agents. Each agent reads / writes the shared state dictionary."""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List

from .config import (CLOSE_CALL_MARGIN, FEATURES, MIN_CONFIDENCE, RAG_TOP_K)
from .data import DatasetStats
from .llm import LLMUnavailable, OllamaLLM
from .model_tool import CropPredictorTool
from .rag import KnowledgeBase
from .state import log
from .validation import validate_measurements

AGRI_WORDS = ("crop", "soil", "irrigat", "fertili", "nutrient", "water", "plant", "farm", "yield",
              "harvest", "pest", "seed", "rain", "nitrogen", "phosph", "potassium", "manure", "compost",
              "drought", "cultivat", "sow", "agricult", "grow", "field", "humidity", "wilting")
RECOMMEND_PATTERN = re.compile(r"which crop|what (crop|to plant|to grow|should i (plant|grow))|recommend|suggest|best crop", re.I)
CITE = re.compile(r"\[S(\d+)\]")


# =============================================================================
# Router (Ollama-based, with a rule-based fallback) + input validation
# =============================================================================
class RouterAgent:
    name = "Router"

    def __init__(self, llm: OllamaLLM, stats: DatasetStats):
        self.llm, self.stats = llm, stats

    def _intent(self, q: str, state: Dict[str, Any]) -> str:
        system = (
            "Classify a farmer's message. Reply ONLY with JSON like {\"intent\": \"...\"} where intent is one of: "
            "\"recommend_request\" (asks which crop to grow or plant), "
            "\"knowledge_question\" (asks about farming: irrigation, fertilizer, soil, a specific crop), "
            "\"out_of_scope\" (anything not about farming)."
        )
        if self.llm.chat_available():
            try:
                intent = str(self.llm.chat_json(system, q).get("intent", "")).strip().lower()
                if intent in ("recommend_request", "knowledge_question", "out_of_scope"):
                    state["route_reason"] = f"intent classified by Ollama ({self.llm.model})"
                    return intent
            except LLMUnavailable:
                pass
        # rule-based fallback
        state["route_reason"] = "intent classified by keyword rules (Ollama unavailable or unclear)"
        ql = q.lower()
        crop_named = any(re.search(rf"\b{c}\b", ql) for c in self.stats.crops())
        if not (crop_named or any(w in ql for w in AGRI_WORDS) or re.search(r"\bph\b", ql)):
            return "out_of_scope"
        return "recommend_request" if RECOMMEND_PATTERN.search(q) else "knowledge_question"

    def run(self, state: Dict[str, Any]) -> None:
        ui = state["user_input"]
        raw_m = ui.get("measurements")
        q = (ui.get("question") or "").strip()
        state["question"] = q

        if raw_m:
            clean, errors, warns = validate_measurements(raw_m, self.stats)
            if errors:
                state.update(route="invalid", validation_errors=errors, route_reason="input validation failed")
                log(state, self.name, f"route=invalid ({len(errors)} error(s))")
                return
            state["measurements"] = clean
            state["warnings"].extend(warns)
            if q and self._intent(q, state) == "out_of_scope":
                state["route"] = "unsupported"
            else:
                state["route"] = "recommend"
                state["route_reason"] = state["route_reason"] or "valid measurements supplied"
        elif q:
            intent = self._intent(q, state)
            if intent == "out_of_scope":
                state["route"] = "unsupported"
            elif intent == "recommend_request":
                state.update(route="invalid", validation_errors=[
                    "To recommend a crop I need all seven measurements: " + ", ".join(FEATURES) + "."])
            else:
                state["route"] = "knowledge"
        else:
            state.update(route="invalid", validation_errors=[
                "Nothing to process: give the seven measurements and/or an agriculture question."],
                route_reason="empty input")
        log(state, self.name, f"route={state['route']} - {state['route_reason']}")


# =============================================================================
# 1. Soil Analysis Agent  (tool: CSV / pandas)
# =============================================================================
class SoilAnalysisAgent:
    name = "SoilAnalysis"

    def __init__(self, stats: DatasetStats):
        self.stats = stats

    @staticmethod
    def _ph_class(ph: float) -> str:
        if ph < 5.5:
            return "strongly acidic"
        if ph < 6.5:
            return "slightly acidic"
        if ph <= 7.5:
            return "neutral"
        return "alkaline"

    def run(self, state: Dict[str, Any]) -> None:
        m = state["measurements"]
        rows, summary = [], []
        for f in FEATURES:
            pct = self.stats.percentile(f, m[f])
            lvl = self.stats.level(f, m[f])
            rows.append({"feature": f, "value": m[f], "percentile": round(pct, 1), "level": lvl})
            summary.append(f"{f} = {m[f]:g} is {lvl} relative to the dataset (percentile {pct:.0f})")
        ph_class = self._ph_class(m["ph"])
        summary.append(f"soil pH {m['ph']:g} is {ph_class}")
        state["soil_analysis"] = {"rows": rows, "ph_class": ph_class, "summary": summary}
        log(state, self.name, f"analysed {len(rows)} measurements with pandas; pH is {ph_class}")


# =============================================================================
# 2. Crop Recommendation Agent  (tool: scikit-learn model)
# =============================================================================
class CropRecommendationAgent:
    name = "CropRecommendation"

    def __init__(self, predictor: CropPredictorTool, stats: DatasetStats):
        self.predictor, self.stats = predictor, stats

    def run(self, state: Dict[str, Any]) -> None:
        pred = self.predictor.predict(state["measurements"])
        state["prediction"] = pred
        crop, p = pred["label"], pred["probability"]
        top = pred["top_k"]
        nn = pred["nearest_neighbor"]
        rationale = [
            f"The trained random-forest classifier predicts '{crop}' with probability {p:.0%} "
            f"(runners-up: " + ", ".join(f"{t['crop']} {t['probability']:.0%}" for t in top[1:]) + ")."
        ]
        rationale.append(
            f"The most similar training row is labelled '{nn['crop']}' (similarity {nn['similarity']:.2f}); "
            + ("this agrees with the prediction." if nn["agrees_with_label"] else "this DISAGREES with the prediction.")
        )
        inside, outside = [], []
        for f in FEATURES:
            lo, hi = self.stats.crop_range(crop, f)
            v = state["measurements"][f]
            (inside if lo <= v <= hi else outside).append((f, v, lo, hi))
        rationale.append(
            f"{len(inside)} of {len(FEATURES)} measurements lie within the typical (5th-95th percentile) "
            f"range of '{crop}' rows in the dataset."
        )
        for f, v, lo, hi in outside:
            rationale.append(f"{f} = {v:g} is outside the typical '{crop}' range {lo:.1f}-{hi:.1f}.")
        state["rationale"] = rationale

        lim = state["limitations"]
        if p < MIN_CONFIDENCE:
            lim.append(f"Low model confidence ({p:.0%}); treat the recommendation as tentative.")
        if len(top) > 1 and (top[0]["probability"] - top[1]["probability"]) < CLOSE_CALL_MARGIN:
            lim.append(f"Close call between '{top[0]['crop']}' and '{top[1]['crop']}'.")
        if not nn["agrees_with_label"]:
            lim.append("The nearest training example belongs to a different crop than the model's prediction.")
        if outside:
            lim.append(f"{len(outside)} input value(s) fall outside the typical range for '{crop}'.")
        log(state, self.name, f"predict -> {crop} ({p:.0%}); nn={nn['crop']}")


# =============================================================================
# 3. Agricultural Knowledge Agent  (tool: Chroma / Ollama embeddings, RAG)
# =============================================================================
class KnowledgeAgent:
    name = "AgriKnowledge"

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def kb_crops(self) -> List[str]:
        return sorted({c.crop for c in self.kb.chunks if c.crop != "general"})

    def _merge(self, groups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Crop-specific passages first, then general ones; drop duplicates and re-number S1..Sn."""
        seen, merged = set(), []
        for g in groups:
            for h in g:
                if h["source"] not in seen:
                    seen.add(h["source"])
                    merged.append({k: v for k, v in h.items() if k != "id"})
        return [{"id": f"S{i}", **h} for i, h in enumerate(merged, start=1)]

    def run(self, state: Dict[str, Any]) -> None:
        if state["route"] == "recommend":
            crop = state["prediction"]["label"]
            query = f"{crop} irrigation water requirement nutrient fertilizer soil pH climate {state['question']}"
            # guarantee the predicted crop's own passage, then add general irrigation / nutrient passages
            specific = self.kb.search(query, k=1, crops=[crop])
            general = self.kb.search(query, k=RAG_TOP_K - 1, crops=["general"])
            hits = self._merge([specific, general]) if specific else []   # no crop evidence -> unsupported
        else:
            q = state["question"].lower()
            named = [c for c in self.kb_crops() if re.search(rf"\b{re.escape(c)}\b", q)][:2]
            if named:
                groups = [self.kb.search(state["question"], k=1, crops=[c]) for c in named]
                groups.append(self.kb.search(state["question"], k=max(1, RAG_TOP_K - len(named)), crops=["general"]))
                hits = self._merge(groups)
            else:
                hits = self.kb.search(state["question"], k=RAG_TOP_K)
        state["retrieved"] = hits
        log(state, self.name, f"retrieved {len(hits)} passage(s) via {self.kb.backend}")
        if not hits:
            state["limitations"].append(
                "No supporting passages were found in the knowledge base, so agronomic advice "
                "(irrigation / nutrients) is withheld rather than guessed.")


# =============================================================================
# Synthesis: drafts / revises the advisory with Ollama (or offline fallback)
# =============================================================================
class WriterAgent:
    name = "Writer"

    def __init__(self, llm: OllamaLLM):
        self.llm = llm

    # ---- prompt pieces ----------------------------------------------------
    @staticmethod
    def _passages(state) -> str:
        return "\n".join(f"[{r['id']}] ({r['source']}) {r['text']}" for r in state["retrieved"])

    @staticmethod
    def facts(state) -> str:
        if state["route"] != "recommend":
            return f"Farmer question: {state['question']}"
        pred, m = state["prediction"], state["measurements"]
        return (
            f"Recommended crop (fixed by a trained model): {pred['label']} (probability {pred['probability']:.2f}).\n"
            f"Other candidates: {', '.join(t['crop'] + ' ' + format(t['probability'], '.2f') for t in pred['top_k'][1:])}.\n"
            f"Farm measurements: " + ", ".join(f"{k}={v:g}" for k, v in m.items()) + ".\n"
            f"Soil summary: " + "; ".join(state["soil_analysis"]["summary"]) + ".\n"
            + (f"Farmer question: {state['question']}" if state["question"] else "")
        )

    def _system(self, state, cite: bool = True) -> str:
        if state["route"] == "recommend":
            crop = state["prediction"]["label"]
            s = (
                "You write short advice for a farmer. RULES: (1) The recommended crop is '" + crop + "'; it was chosen by a "
                "trained model - never suggest a different main crop. (2) Use ONLY the given facts and numbered passages. "
                "If they do not cover something (for example an exact irrigation schedule or fertilizer quantity) write "
                "'not covered by the knowledge base' - never invent numbers. "
                "(3) Format: three short sections titled 'Irrigation', 'Nutrient management' and 'Cautions', "
                "each with 2-3 bullet points, under 200 words in total."
            )
        else:
            s = (
                "You answer a farmer's question. Use ONLY the numbered passages. If they do not answer the question, say so. "
                "Never invent numbers. Keep it under 150 words."
            )
        if cite:
            s += " (4) After every statement taken from a passage, cite its id in square brackets, e.g. [S1]."
        else:
            s += " (4) Do NOT include any passage ids or citations."
        return s

    # ---- offline fallback -------------------------------------------------
    @staticmethod
    def _offline(state, cite: bool = True) -> str:
        lines = ["Guidance assembled directly from the knowledge base (language model unavailable):"]
        for r in state["retrieved"]:
            lines.append(f"- {r['text']}" + (f" [{r['id']}]" if cite else ""))
        return "\n".join(lines)

    # ---- public -----------------------------------------------------------
    def draft(self, state: Dict[str, Any]) -> None:
        # In --demo-revision mode the FIRST draft deliberately omits citations, so the
        # reviewer rejects it and the single allowed revision is demonstrated.
        cite = not (state["demo_revision"] and state["retry_count"] == 0)
        user = f"FACTS:\n{self.facts(state)}\n\nPASSAGES:\n{self._passages(state)}"
        try:
            state["draft"] = self.llm.chat(self._system(state, cite), user, temperature=0.2)
            log(state, self.name, f"draft written by Ollama ({self.llm.model}), citations={'on' if cite else 'OFF (demo)'}")
        except LLMUnavailable as exc:
            state["draft"] = self._offline(state, cite)
            self._note_offline(state, exc)
            log(state, self.name, "draft assembled offline from retrieved passages")

    def revise(self, state: Dict[str, Any]) -> None:
        issues = "\n".join(f"- {i}" for i in state["review"]["issues"])
        user = (f"FACTS:\n{self.facts(state)}\n\nPASSAGES:\n{self._passages(state)}\n\n"
                f"PREVIOUS DRAFT:\n{state['draft']}\n\nREVIEWER ISSUES TO FIX:\n{issues}\n\n"
                "Rewrite the draft and fix every issue.")
        try:
            state["draft"] = self.llm.chat(self._system(state, cite=True), user, temperature=0.1)
            log(state, self.name, "draft revised by Ollama after review feedback")
        except LLMUnavailable as exc:
            state["draft"] = self._offline(state, cite=True)
            self._note_offline(state, exc)
            log(state, self.name, "draft re-assembled offline after review feedback")

    @staticmethod
    def _note_offline(state, exc) -> None:
        msg = "Ollama was unavailable, so the advisory text was assembled directly from retrieved passages."
        if msg not in state["limitations"]:
            state["limitations"].append(msg)


# =============================================================================
# 4. Validation Agent  (tool: rule checks + Ollama reviewer)
# =============================================================================
class ValidationAgent:
    name = "Validation"

    def __init__(self, llm: OllamaLLM, stats: DatasetStats):
        self.llm, self.stats = llm, stats

    @staticmethod
    def _numbers(text: str) -> set:
        return set(re.findall(r"\d+(?:\.\d+)?", text))

    def rule_checks(self, state: Dict[str, Any]) -> List[str]:
        issues: List[str] = []
        draft = state["draft"] or ""
        if not draft.strip():
            return ["The draft is empty."]

        # 1. citations must exist and point at retrieved passages
        cited = set(CITE.findall(draft))
        valid = {r["id"][1:] for r in state["retrieved"]}
        if not cited:
            issues.append("The draft has no [S#] citations to the retrieved evidence.")
        unknown = cited - valid
        if unknown:
            issues.append("The draft cites passages that were not retrieved: " + ", ".join(f"[S{u}]" for u in sorted(unknown)) + ".")

        # 2. no unrelated crop recommended (the crop is fixed by the model)
        if state["route"] == "recommend":
            allowed = {t["crop"] for t in state["prediction"]["top_k"]}
            low = draft.lower()
            for c in self.stats.crops():
                if c not in allowed and re.search(rf"\b{re.escape(c)}\b", low):
                    issues.append(f"The draft mentions '{c}', which is not among the model's candidate crops.")

        # 3. numbers must come from the evidence or the supplied facts
        allowed_nums = self._numbers(" ".join(r["text"] for r in state["retrieved"]) + " " + WriterAgent.facts(state))
        body = CITE.sub("", draft)
        extra = sorted(n for n in self._numbers(body) if n not in allowed_nums and n not in {"1", "2", "3"})
        if extra:
            issues.append("The draft contains numbers not found in the evidence or model output: " + ", ".join(extra) + ".")
        return issues

    def review(self, state: Dict[str, Any]) -> None:
        issues = self.rule_checks(state)
        by = "rules"
        if not issues and self.llm.chat_available():
            system = (
                "You are a strict reviewer of farm advice. Compare the DRAFT with the PASSAGES and FACTS. "
                "Reply ONLY with JSON: {\"verdict\": \"approve\" or \"revise\", \"issues\": [short strings]}. "
                "Use \"revise\" only for claims not supported by the passages/facts, or advice contradicting the chosen crop."
            )
            user = f"FACTS:\n{WriterAgent.facts(state)}\n\nPASSAGES:\n{WriterAgent._passages(state)}\n\nDRAFT:\n{state['draft']}"
            try:
                out = self.llm.chat_json(system, user)
                if str(out.get("verdict", "")).lower() == "revise":
                    llm_issues = [str(i) for i in out.get("issues", []) if str(i).strip()]
                    issues = [f"LLM reviewer: {i}" for i in llm_issues] or ["LLM reviewer asked for a revision."]
                by = f"rules+ollama ({self.llm.model})"
            except LLMUnavailable:
                by = "rules (Ollama reviewer unavailable)"
        state["review"] = {"verdict": "revise" if issues else "approve", "issues": issues, "by": by}
        log(state, self.name, f"verdict={state['review']['verdict']} by {by}; issues={len(issues)}")
