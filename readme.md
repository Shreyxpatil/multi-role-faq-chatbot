````markdown
# Multi-Role FAQ Chatbot with Dense and Lexical Retrieval

## Overview

This project implements a **multi-role FAQ chatbot API** using a **hybrid retrieval system** that combines **dense embeddings** and **lexical search**. It supports separate endpoints for different user roles such as **student**, **faculty**, **coordinator**, etc.

The backend uses the pretrained **`all-mpnet-base-v2`** sentence embedding model to encode questions and answers, enabling accurate retrieval from role-specific knowledge bases.

---

## Project Structure

| File / Folder           | Description                                                       |
|-------------------------|-------------------------------------------------------------------|
| `api_strict.py`         | Flask API server with role-based endpoints and hybrid FAQ retrieval |
| `all-mpnet-base-v2/`    | Pretrained MPNet embedding model (or path via env var)           |
| `faqs_split_roles/`     | JSON FAQ files split by roles (e.g., `faqs_student.json`)        |
| `train_script.py`       | Script to fine-tune the embedding model (TPU supported)          |
| `config.json`           | Embedding model configuration (pooling mode, etc.)               |

---

## Features

- Role-specific FAQ knowledge bases  
- Hybrid search → **Dense (FAISS)** + **Lexical (TF-IDF)**  
- Uses MPNet (`all-mpnet-base-v2`) for sentence embeddings  
- Fallback response when no answer meets threshold  
- TPU-accelerated fine-tuning script  
- Flask API with CORS support  

---

## Installation

### 1) Clone the repository
```bash
git clone https://github.com/Shreyxpatil/multi-role-faq-chatbot.git
cd multi-role-faq-chatbot
````

> If your project lives elsewhere (e.g., `E:\GateTutor_Questions\Projects\chatbot updated`), `cd` there instead:
>
> **Windows (PowerShell):**
>
> ```powershell
> cd "E:\GateTutor_Questions\Projects\chatbot updated"
> ```
>
> **Linux/macOS:**
>
> ```bash
> cd "/path/to/GateTutor_Questions/Projects/chatbot updated"
> ```

### 2) (Recommended) Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install torch torchvision torchaudio transformers flask flask-cors faiss-cpu scikit-learn torch-xla tqdm
```

### 4) Place or point to the pretrained model

Either **place** the `all-mpnet-base-v2` folder at the project root, **or** set an environment variable:

**Windows (PowerShell)**

```powershell
$env:RETRIEVER_MODEL_PATH="E:\models\all-mpnet-base-v2"
```

**Linux/macOS**

```bash
export RETRIEVER_MODEL_PATH="/models/all-mpnet-base-v2"
```

### 5) Add FAQ JSON files

Place files like `faqs_student.json`, `faqs_faculty.json` in:

```
faqs_split_roles/
```

---

## Running the API

Start the Flask server:

**If your file path has spaces, quote it.**

**Windows (PowerShell)**

```powershell
python "E:\GateTutor_Questions\Projects\chatbot updated\api_strict.py"
```

**From repo root**

```bash
python api_strict.py
```

By default the server runs at:

```
http://0.0.0.0:5000
```

---

## API Endpoints

| Method | Endpoint          | Description                        |
| -----: | ----------------- | ---------------------------------- |
|  `GET` | `/health`         | Health check                       |
| `POST` | `/api/<role>/ask` | Ask a question for a specific role |

### Example Request

```http
POST /api/student/ask
Content-Type: application/json

{
  "query": "How do I reset my password?"
}
```

### Example Response

```json
{
  "answer": "To reset your password, go to the settings page and click 'Reset Password'.",
  "role": "student"
}
```

**Fallback (no answer meets threshold):**

```json
{
  "answer": "Sorry, I couldn't find a relevant answer. Please contact support.",
  "role": "student"
}
```

---

## Training the Embedding Model

Fine-tune MPNet using TPU:

```bash
python "E:\GateTutor_Questions\Projects\chatbot updated\all-mpnet-base-v2\train_script.py" \
  --steps 2000 --batch_size 64 --model microsoft/mpnet-base data_config.json output_dir
```

* `data_config.json` → Defines dataset paths & weights
* `output_dir` → Folder where the fine-tuned model will be saved

> **Note:** Configure TPU environment variables if training on TPU.

---

## Configuration

* `config.json` (inside the model folder) controls pooling modes & architecture.
* Retrieval thresholds and weights (e.g., `THRESHOLD`, `ALPHA`) are defined in `api_strict.py`.

---

## License

This project uses the pretrained **MPNet** model from Hugging Face.
Please review and comply with the respective licenses before commercial use.

---

For more details, see: `all-mpnet-base-v2/README.md`

```
```
