"""Step 8: build the final advisory report (candidate crop, rationale, evidence, limitations)."""
from __future__ import annotations
import textwrap
from typing import Any, Dict, List

STANDARD_LIMITATIONS = [
    "No live weather forecast is used: temperature, humidity and rainfall are the single values you supplied.",
    "The advisory text is limited to the small knowledge corpus; exact irrigation schedules or fertilizer "
    "quantities are only given if the retrieved passages state them.",
    "This is decision support, not a substitute for a local soil test or an agronomist's advice.",
]


def _wrap(text: str, indent: str = "   ") -> str:
    return textwrap.fill(text, width=100, initial_indent=indent, subsequent_indent=indent)


def build_report(state: Dict[str, Any], model_accuracy: float | None = None) -> Dict[str, Any]:
    route = state["route"]
    lim: List[str] = list(state["limitations"])
    final: Dict[str, Any] = {"route": route, "review_status": state["review_status"],
                             "retry_count": state["retry_count"]}

    if route == "invalid":
        final["message"] = ("I can't process this input safely, so no recommendation was made.\n"
                            + "\n".join(f" - {e}" for e in state["validation_errors"])
                            + "\nPlease provide N, P, K, temperature (C), humidity (%), ph and rainfall (mm).")
    elif route == "unsupported":
        final["message"] = ("This assistant only handles crop selection and agronomy questions "
                            "(soil, irrigation, nutrients, specific crops). No prediction was made.")
    else:
        if state["review"] and state["review"]["verdict"] == "revise":
            lim.append("Reviewer issues that remained after the single allowed revision: "
                       + "; ".join(state["review"]["issues"]))
        lim += [w for w in state["warnings"] if w not in lim]
        if route == "recommend":
            pred = state["prediction"]
            final.update(candidate_crop=pred["label"], confidence=pred["probability"],
                         alternatives=pred["top_k"][1:], model_rationale=state["rationale"],
                         soil_analysis=state["soil_analysis"]["summary"])
            if model_accuracy is not None:
                lim.insert(0, f"The crop label comes from a random-forest model trained on the public Kaggle "
                              f"crop-recommendation data; held-out test accuracy was {model_accuracy:.1%}, "
                              "and real fields can differ from that dataset.")
        final["advisory"] = state["draft"] if state["retrieved"] else None
        final["evidence"] = state["retrieved"]
        lim += STANDARD_LIMITATIONS
    final["limitations"] = lim
    final["text"] = format_text(final, state)
    return final


def format_text(f: Dict[str, Any], state: Dict[str, Any]) -> str:
    out = ["=" * 100, "CROP ADVISORY REPORT", "=" * 100]
    if "message" in f:
        out += [f["message"], ""]
        return "\n".join(out)

    out.append(f"Route: {f['route']}   |   Review status: {f['review_status']}   |   Revisions used: {f['retry_count']}")
    if f["route"] == "recommend":
        out += ["", f"CANDIDATE CROP: {f['candidate_crop'].upper()}   (model probability {f['confidence']:.0%})"]
        if f["alternatives"]:
            out.append("Alternatives: " + ", ".join(f"{a['crop']} ({a['probability']:.0%})" for a in f["alternatives"]))
        out += ["", "MODEL-BASED RATIONALE"] + [_wrap("- " + r) for r in f["model_rationale"]]
        out += ["", "SOIL ANALYSIS (relative to the training dataset)"] + [_wrap("- " + s) for s in f["soil_analysis"]]
    out += ["", "ADVISORY" if f["route"] == "recommend" else "ANSWER"]
    out.append(f["advisory"] if f["advisory"] else _wrap("Withheld: no supporting evidence was found in the knowledge base."))
    out += ["", "RETRIEVED SOURCE EVIDENCE"]
    if f["evidence"]:
        for r in f["evidence"]:
            out.append(_wrap(f"[{r['id']}] {r['source']} (score {r['score']}): {r['text']}", indent=""))
    else:
        out.append("   (none)")
    out += ["", "LIMITATIONS"] + [_wrap("- " + l) for l in f["limitations"]]
    return "\n".join(out)
