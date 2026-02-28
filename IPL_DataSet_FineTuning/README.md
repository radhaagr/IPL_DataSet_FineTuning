
## Dataset
**IPL season**.
- Use this dataset only for training and evaluation.
- Allowed fine-tuning methods: **Supervised Fine-Tuning (SFT)** or **LoRA/QLoRA**.

---

## Model Choice & Training Environment
- Use **LLM**

---

## Objective
Enable a user to ask a **cricket statistics question** about the provided IPL season and have the system answer it.  


## IPL Sample Data
The IPL sample data should be in data folder. But since the data is Confidential, I can not upload it here. 
---

## Tasks

1. **Fine-Tuning (SFT or LoRA)**  
   - Fine-tune your chosen model on the provided IPL Q&A dataset.  
   - Must run reproducibly on CPU or single GPU.  
   - Pushed the fine-tuned model to a **private Hugging Face repo**.  

2. **Inference Service (FastAPI)**  
   - Endpoints: `/infer`, `/healthz`, `/readyz`.  
   - Load model from the **private HF repo** using an access token (`HF_TOKEN` env var).  
   - Provide a `Dockerfile`.

3. **Multi-Agent Orchestration**  
   - Example:  
     - **RetrieverAgent**: computes cricket stats (e.g., runs in last N overs).  
     - **AnalystAgent**: queries fine-tuned model with the stats as context.  
   - Deliverable: `agents/multi_agent.py`.


6. **Model Monitoring (Document)**  
   - In `MODEL_MONITORING.md`, describe how to monitor accuracy & relevance in production.  
   - Cover: eval set usage, feedback collection, drift detection.  


