
## Dataset
**IPL season**.
- Use this dataset only for training and evaluation.
- Allowed fine-tuning methods: **Supervised Fine-Tuning (SFT)** or **LoRA/QLoRA**.

---

## Model Choice & Training Environment
- You may choose either a **small language model (SLM)** or a **larger LLM**, depending on your hardware.  
- **CPU or single GPU is absolutely fine** for this test.  
- If you do have access to a **GPU cluster**, feel free to use it—that’s a bonus, not a requirement.  
- Training runs do not need to be heavy—keep them small and reproducible.  

---

## Objective
Enable a user to ask a **cricket statistics question** about the provided IPL season and have the system answer it.  
We understand the answers won’t always be perfect or 100% accurate or 100% complete — what we care about is your **thinking, approach, and ability to write production-grade code**.

## IPL Sample Data
The IPL sample data is provided in data folder. This data is property of Xansr and can only be used for this test purposes. Any other usage of the data whatsoever will be voilation of the test agreement and subject to prosecution.

---

## Tasks

1. **Fine-Tuning (SFT or LoRA)**  
   - Fine-tune your chosen model on the provided IPL Q&A dataset.  
   - Must run reproducibly on CPU or single GPU.  
   - Push your fine-tuned model to a **private Hugging Face repo**.  

2. **Inference Service (FastAPI)**  
   - Endpoints: `/infer`, `/healthz`, `/readyz`.  
   - Load your model from the **private HF repo** using an access token (`HF_TOKEN` env var).  
   - Provide a `Dockerfile`.

3. **Multi-Agent Orchestration**  
   - Example:  
     - **RetrieverAgent**: computes cricket stats (e.g., runs in last N overs).  
     - **AnalystAgent**: queries fine-tuned model with the stats as context.  
   - Deliverable: `agents/multi_agent.py`.

4. **Deployment (AKS/Helm)**  
   - Helm chart + manifests.  
   - Include GPU resource hints.  
   - Inject `HF_TOKEN` as env var.

5. **Observability & Testing**  
   - Log latency per request.  
   - Provide one integration test for `/infer`.  
   - Document performance considerations.

6. **Model Monitoring (Document)**  
   - In `MODEL_MONITORING.md`, describe how you would monitor accuracy & relevance in production.  
   - Cover: eval set usage, feedback collection, drift detection.  
   - No need to implement.

7. **Accuracy Scaling Strategy (Document)**  
   - In this `README.md`, include a short plan for how you would **increase accuracy if given scaled GPU compute**.  
   - Consider: larger base model, longer context, more/cleaner training data, better LoRA hyperparameters, improved retrieval/tooling, etc.  
   - No need to implement—just outline your approach.

---

## Deliverables
- `scripts/train_sft.py`  
- `service/app.py` + `Dockerfile`  
- `agents/multi_agent.py`  
- `deploy/helm/`  
- `tests/test_inference.py`  
- `MODEL_MONITORING.md`  
- This `README.md` (instructions + your decisions/plan below)

---

## Rubric Benchmarks
You will be benchmarked and evaluated on the following criteria.
- **Fine-Tuning (SFT/LoRA, HF repo publish)** — 45%  
- **Inference Service (FastAPI + HF repo loading)** — 25%  
- **Multi-Agent Orchestration** — 15%  
- **Deployment (AKS/Helm)** — 5%  
- **Observability & Testing** — 5%  
- **Accuracy Scaling Strategy (documented)** — 5%  

---

# Candidate Decisions & Plans (fill in)

- **Base Model Chosen (SLM or LLM):** …  
- **SFT vs LoRA (and why):** …  
- **Training Setup (CPU / GPU / Cluster):** …  
- **HF Private Repo (name):** …  
- **Inference Approach (load & cache, batching):** …  
- **Multi-Agent Design (retriever/tooling → analyst/model):** …  
- **Observability Notes (latency logs, metrics):** …  

## Accuracy Scaling Strategy
- **Model capacity:** …  
- **Data & training:** …  
- **Retrieval/tooling:** …  
- **Inference-time tricks:** …  
- **Evaluation loop:** …  
