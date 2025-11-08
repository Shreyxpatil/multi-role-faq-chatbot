````markdown
# Multi-Role FAQ Chatbot with Dense and Lexical Retrieval

## Overview

This project implements a **multi-role FAQ chatbot API** using a **hybrid retrieval system** that combines **dense embeddings** and **lexical search**. It supports separate endpoints for different user roles such as **student**, **faculty**, **coordinator**, etc.

The backend uses the pretrained **`all-mpnet-base-v2`** sentence embedding model to encode questions and answers, enabling accurate retrieval from role-specific knowledge bases.

---

## Project Structure

| File / Folder | Description |
|--------------|-------------|
| `api_strict.py` | Flask API server with role-based endpoints and hybrid FAQ retrieval |
| `all-mpnet-base-v2/` | Folder containing the pretrained MPNet embedding model |
| `faqs_split_roles/` | JSON FAQ files split by roles (e.g., `faqs_student.json`) |
| `train_script.py` | Script to fine-tune the embedding model (TPU supported) |
| `config.json` | Embedding model configuration (pooling mode, etc.) |

---

## Features

✅ Role-specific FAQ knowledge bases  
✅ Hybrid search → **Dense (FAISS)** + **Lexical (TF-IDF)**  
✅ Uses MPNet (`all-mpnet-base-v2`) for sentence embeddings  
✅ Fallback response when no answer meets threshold  
✅ TPU-accelerated fine-tuning script  
✅ Flask API with CORS support  

---

## Installation

### 1. Clone the repository
```bash
git clone <repo-url>
cd <project-folder>
````

### 2. Install dependencies

```bash
pip install torch torchvision torchaudio transformers flask flask-cors faiss-cpu scikit-learn torch-xla tqdm
```

### 3. Place the pretrained model

Either:

* Put the `all-mpnet-base-v2` folder in the project root
  **OR**
* Set an environment variable:

```bash
export RETRIEVER_MODEL_PATH="path/to/model"
```

### 4. Add FAQ JSON files

Store files like `faqs_student.json`, `faqs_faculty.json` in:

```
faqs_split_roles/
```

---

## Running the API

Start the Flask server:

```bash
python e:\GateTutor_Questions\Projects\chatbot updated\api_strict.py
```

Server will run at:

```
http://0.0.0.0:5000
```

---

## API Endpoints

| Method | Endpoint          | Description                        |
| ------ | ----------------- | ---------------------------------- |
| `GET`  | `/health`         | Check if server is running         |
| `POST` | `/api/<role>/ask` | Ask a question for a specific role |

### Example Request

```
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

If no answer passes the threshold:

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
python e:\GateTutor_Questions\Projects\chatbot updated\all-mpnet-base-v2\train_script.py --steps 2000 --batch_size 64 --model microsoft/mpnet-base data_config.json output_dir
```

* `data_config.json` → Defines dataset paths & weights
* `output_dir` → Folder where trained model will be saved

> **Note:** TPU environment variables must be configured if training on TPU.

---

## Configuration

* `config.json` (in model folder) controls pooling mode & architecture
* Retrieval thresholds (e.g., `THRESHOLD`, `ALPHA`) are inside `api_strict.py`

---

## License

This project uses the pretrained **MPNet model** from Hugging Face.
Refer to their license terms before commercial use.

---

For additional details, see:
`all-mpnet-base-v2/README.md`

```

Let me know if you want a **downloadable README file**, **badges**, or a **"How it works" diagram**.
```
