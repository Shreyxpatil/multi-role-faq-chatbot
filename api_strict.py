# api_strict.py — multi-role FAQ API (strict retrieval, role-specific endpoints)

import os, re, json, logging
from typing import List, Dict, Tuple
from pathlib import Path

import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

# ---------------- Config ----------------
BASE_FAQ_DIR = Path(r"E:\GateTutor_Questions\Projects\chatbot updated\faqs_split_roles")

ROLE_PATHS = {
    "student": BASE_FAQ_DIR / "faqs_student.json",
    "faculty": BASE_FAQ_DIR / "faqs_faculty.json",
    "coordinator": BASE_FAQ_DIR / "faqs_coordinator.json",
    "controller": BASE_FAQ_DIR / "faqs_controller.json",
    "tpo": BASE_FAQ_DIR / "faqs_tpo.json",
    "corporate": BASE_FAQ_DIR / "faqs_corporate.json",
    "admin": BASE_FAQ_DIR / "faqs_admin.json",
}

MODEL_PATH = Path(os.environ.get("RETRIEVER_MODEL_PATH", "./all-mpnet-base-v2"))
TOP_K_DENSE = 10
TOP_K_FINAL = 3
ALPHA = 0.75
THRESHOLD = 0.45
HOST = "0.0.0.0"
PORT = 5000
DEBUG = False

FALLBACK_MESSAGE = "I am sorry, I am not able to answer. Please try to rephrase your question."

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("gt-faq-strict")

# ---------------- Flask ----------------
app = Flask(__name__)
CORS(app)

# ---------------- Globals ----------------
class RoleState:
    def __init__(self):
        self.kb_items: List[Dict] = []
        self.questions: List[str] = []
        self.answers: List[str] = []
        self.docs: List[str] = []
        self.q_index = None
        self.qa_index = None
        self.tfidf = None
        self.tfidf_matrix = None

role_states: Dict[str, RoleState] = {}

# Dense encoder
tokenizer = None
encoder = None
device = None
emb_dim = None

# ---------------- Utils ----------------
def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s

def build_docs(qs: List[str], ans: List[str]) -> List[str]:
    return [f"{normalize_text(q)}\n\n{normalize_text(a)}" for q, a in zip(qs, ans)]

# ---------------- Encoder ----------------
class LocalEncoder:
    def __init__(self, model_path: Path):
        global tokenizer, encoder, device, emb_dim
        if tokenizer is None or encoder is None:
            log.info(f"Loading local encoder from: {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
            encoder = AutoModel.from_pretrained(str(model_path), local_files_only=True)
            encoder.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoder.to(device)
            emb_dim = encoder.config.hidden_size
            log.info(f"Encoder ready | hidden_size={emb_dim} | device={device}")

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        batch = 64
        out_vecs = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i+batch]
            tok = tokenizer(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt")
            tok = {k: v.to(device) for k, v in tok.items()}
            out = encoder(**tok)
            hidden = out.last_hidden_state
            mask = tok["attention_mask"].unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            emb = (summed / counts).cpu().numpy().astype("float32")
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norms, 1e-9, None)
            out_vecs.append(emb)
        return np.vstack(out_vecs)

# ---------------- Build Indices ----------------
def build_dense_indices(state: RoleState):
    enc = LocalEncoder(MODEL_PATH)
    q_vecs = enc.encode(state.questions)
    qa_vecs = enc.encode(state.docs)
    state.q_index = faiss.IndexFlatIP(q_vecs.shape[1])
    state.q_index.add(q_vecs)
    state.qa_index = faiss.IndexFlatIP(qa_vecs.shape[1])
    state.qa_index.add(qa_vecs)

def build_lexical_index(state: RoleState):
    state.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, lowercase=True, norm="l2")
    state.tfidf_matrix = state.tfidf.fit_transform([normalize_text(d) for d in state.docs])

# ---------------- Search ----------------
def dense_search(vecs, index, top_k):
    scores, idxs = index.search(vecs, top_k)
    return scores[0], idxs[0]

def lexical_search(query, state, top_k):
    q_vec = state.tfidf.transform([normalize_text(query)])
    q_vec = sk_normalize(q_vec, norm="l2", copy=False)
    sims = (state.tfidf_matrix @ q_vec.T).toarray().ravel()
    if top_k >= len(sims):
        idxs = np.argsort(-sims)
    else:
        idxs = np.argpartition(-sims, top_k)[:top_k]
        idxs = idxs[np.argsort(-sims[idxs])]
    return sims[idxs], idxs

def hybrid_search(query: str, state: RoleState) -> Tuple[List[int], List[float]]:
    enc = LocalEncoder(MODEL_PATH)
    qv = enc.encode([normalize_text(query)])

    q_scores, q_idxs = dense_search(qv, state.q_index, TOP_K_DENSE)
    qa_scores, qa_idxs = dense_search(qv, state.qa_index, TOP_K_DENSE)
    lex_scores, lex_idxs = lexical_search(query, state, TOP_K_DENSE)

    cand = set(q_idxs.tolist()) | set(qa_idxs.tolist()) | set(lex_idxs.tolist())
    cand = [i for i in cand if 0 <= i < len(state.kb_items)]
    if not cand:
        return [], []

    cand_scores = []
    for i in cand:
        ds = 0.0
        if i in q_idxs:
            ds = max(ds, float(q_scores[np.where(q_idxs == i)[0][0]]))
        if i in qa_idxs:
            ds = max(ds, float(qa_scores[np.where(qa_idxs == i)[0][0]]))
        ls = 0.0
        if i in lex_idxs:
            ls = float(lex_scores[np.where(lex_idxs == i)[0][0]])
        hs = ALPHA * ds + (1.0 - ALPHA) * ls
        cand_scores.append((i, ds, ls, hs))

    cand_scores.sort(key=lambda x: x[3], reverse=True)
    top = cand_scores[:TOP_K_FINAL]
    idxs = [i for (i, _, _, _) in top]
    scores = [h for (_, _, _, h) in top]
    return idxs, scores

# ---------------- KB ----------------
def load_kb(path: Path) -> RoleState:
    state = RoleState()
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = []
    for i, r in enumerate(raw):
        q = normalize_text(r.get("question") or r.get("title") or "")
        a = normalize_text(r.get("answer") or r.get("a") or "")
        if not q or not a:
            continue
        items.append({"id": f"faq:{i}", "title": q, "answer": a, "source": path.name})
    state.kb_items = items
    state.questions = [it["title"] for it in items]
    state.answers   = [it["answer"] for it in items]
    state.docs      = build_docs(state.questions, state.answers)
    build_dense_indices(state)
    build_lexical_index(state)
    return state

def bootstrap():
    for role, path in ROLE_PATHS.items():
        if not path.exists():
            log.warning("FAQ file not found for role %s: %s", role, path)
            continue
        log.info("Loading KB for role: %s", role)
        role_states[role] = load_kb(path)
    log.info("All role knowledge bases loaded: %s", list(role_states.keys()))

# ---------------- Routes ----------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "roles": list(role_states.keys())
    })

def make_role_endpoint(role: str):
    route = f"/api/{role}/ask"

    def _role_ask():
        if role not in role_states:
            return jsonify({"answer": f"Role {role} not available."}), 400
        state = role_states[role]
        if not request.is_json:
            return jsonify({"answer": "Invalid request (JSON required)."}), 400
        data = request.get_json(silent=True) or {}
        q = normalize_text(data.get("query") or "")
        if not q:
            return jsonify({"answer": "Please provide a question."}), 400

        idxs, scores = hybrid_search(q, state)
        if not idxs or scores[0] < THRESHOLD:
            return jsonify({"answer": FALLBACK_MESSAGE, "role": role})
        i = idxs[0]
        return jsonify({"answer": state.kb_items[i]["answer"], "role": role})

    # ✅ Unique endpoint name per role
    app.add_url_rule(route, endpoint=f"{role}_ask", view_func=_role_ask, methods=["POST"])

# Create one endpoint per role
for r in ROLE_PATHS:
    make_role_endpoint(r)

# ---------------- Main ----------------
if __name__ == "__main__":
    bootstrap()
    app.run(host=HOST, port=PORT, debug=DEBUG)
