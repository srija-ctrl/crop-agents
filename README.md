# Multi-Agent Intelligent Crop Management System

Orchestrator + 4 specialised agents + Ollama, over the Kaggle *Crop Recommendation* dataset.

```
Farmer input (N, P, K, temperature, humidity, ph, rainfall  [+ optional question])
        |
   Orchestrator  (Router: Ollama intent + input validation, shared state, max 1 revision)
        |
 1. Soil Analysis Agent ............ pandas over the CSV (levels vs dataset, pH class)
 2. Crop Recommendation Agent ...... RandomForest tool -> label, probabilities, nearest-neighbour similarity
 3. Agricultural Knowledge Agent ... RAG over data/knowledge/*.txt (Chroma + Ollama embeddings, TF-IDF fallback)
 4. Validation Agent ............... rule checks + Ollama reviewer (citations, unsupported numbers, wrong crop)
        |
   Advisory report: candidate crop, model rationale, source evidence, limitations
```

## 1. Setup on Debian

```bash
sudo apt update && sudo apt install -y python3 python3-venv python3-pip
cd crop-agents
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Ollama (you already have 0.30.0):

```bash
ollama pull llama3.2:3b          # chat model (routing / drafting / review)
ollama pull nomic-embed-text     # embedding model for the RAG store
curl http://localhost:11434      # should print "Ollama is running"
# if not running:  sudo systemctl start ollama     (or: ollama serve)
```

Want a different chat model? `export OLLAMA_MODEL=llama3.1:8b` (any model you have pulled).
If the configured model is missing, the code automatically uses the first chat model it finds.

## 2. Get the dataset (required)

Download `Crop_recommendation.csv` from
<https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset>
and put it at `data/Crop_recommendation.csv`.

(Only to smoke-test before downloading: `python scripts/make_sample_data.py` then
`export CROP_CSV=data/Crop_recommendation_SYNTHETIC.csv`. It is synthetic - never submit results from it.)

## 3. Run

```bash
python train.py            # steps 1-2: inspect data, split (seed 42), train, save model + metrics
python main.py --demo      # steps 3-8: 6 scenarios incl. invalid input and a forced revision
python main.py --N 90 --P 42 --K 43 --temperature 20.9 --humidity 82 --ph 6.5 --rainfall 203 --trace
python main.py --question "How should I irrigate banana?"      # knowledge-only route
python main.py --demo --no-llm                                  # works without Ollama
pytest -q                                                       # tests (offline, no Ollama needed)
```

In VS Code: open the folder, select the `.venv` interpreter, then use **Run and Debug**
(configs for train / demo / demo-without-Ollama / single input are in `.vscode/launch.json`).

Outputs go to `outputs/` (`last_report.txt`, `last_run.json` = full shared state, `metrics.json`).

## 4. Where each assignment step lives

| Step | File |
|---|---|
| 1 read CSV, columns, missing values, label distribution | `crop_agents/data.py` (`inspect_dataset`), `train.py` |
| 2 fixed-seed split + lightweight classifier | `crop_agents/model_tool.py` (`train_and_save`) |
| 3 predictor as a tool (label, probabilities, NN similarity) | `crop_agents/model_tool.py` (`CropPredictorTool`) |
| 4 text-only RAG corpus (CSV rows are NOT documents) | `crop_agents/rag.py`, `data/knowledge/agri_knowledge.txt` |
| 5 Ollama for routing / synthesis / review; prediction from the tool only | `crop_agents/llm.py`, `crop_agents/agents.py` |
| 6 shared state dict | `crop_agents/state.py` |
| 7 conditional routing, <=1 revision, safe responses | `crop_agents/orchestrator.py`, `crop_agents/validation.py` |
| 8 candidate crop, rationale, evidence, limitations | `crop_agents/report.py` |

Routes: `recommend`, `knowledge` (question only), `invalid` (bad/missing numbers), `unsupported`
(non-farming question). If no supporting passage is retrieved, advisory text is withheld instead of guessed.

## 5. Use your own RAG corpus

Drop `.txt` files into `data/knowledge/`. Best format: `## crop-name` headings (lower-case, matching
dataset labels) followed by prose; plain paragraph files also work. Then run `python main.py --rebuild-kb --demo`.

## Notes

* Real embeddings + chat need Ollama running. Without it, everything degrades gracefully (TF-IDF retrieval,
  passage-based drafts) and the report says so under *Limitations*.
* Thresholds (confidence, retrieval score, plausible ranges) are in `crop_agents/config.py`.
# crop-agents
