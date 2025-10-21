# Suicide Intention Detection — RoBERTa-CNN Model

This repository contains the source code for the **Suicide Intention Detection** project, developed as part of research prior to the University of Denver Computer Science application. The project implements a hybrid deep learning model combining RoBERTa (Transformer) and CNN for binary text classification (suicide / non-suicide).

---

## 📂 File Structure

- `RoBERTA_CNN-Training.final.py` — Main training and evaluation script
- `README.md` — Project documentation

---

## ⚙️ Environment Setup

Ensure the following packages are installed:

- Python 3.8+
# FINAL — Suicide Intention Detection (RoBERTa + CNN)

This folder contains the final artifacts and documentation for the Suicide Intention Detection experiments that use a hybrid RoBERTa + CNN architecture for binary classification (suicidal / non-suicidal text).

The goal of this work is to explore automated methods for detecting high-risk language in text so that researchers and practitioners can build safer, faster triage systems, while carefully considering ethical and privacy constraints.

Why this matters (public mental health):
- Detecting signals of suicide risk in text can help public health teams and crisis support organizations identify at-risk individuals earlier and allocate resources more effectively.
- Automated screening tools can assist human clinicians and moderators by prioritizing cases that need immediate attention, reducing response time in online communities.
- Research in this area furthers our understanding of language markers of distress and can inform prevention strategies at population scale.

> Important: models are aides — not replacements for trained clinicians. Deployment must include safeguards, human review, and clear pathways to help.

---

## Files in this folder

- `RoBERTa CNN-traning_final.py` — Final training & evaluation script used for reported results (entrypoint for training experiments).
- `README.md` — (this file) project summary, run instructions and upload guidance.
- (Other artifacts) — model config, saved weights or small exports may be present; large weights should be stored using Git LFS or WandB Artifacts rather than committed directly.

---

## Quick start

1. Create & activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (adjust the list in the repo `requirements.txt` as needed):

```bash
pip install -r ../requirements.txt
```

3. Train / evaluate (example):

```bash
python "RoBERTa CNN-traning_final.py" --config configs/final_config.yaml
```

Replace `--config` with the arguments or configuration style used by the script. Open the script to see available CLI flags and default file paths.

---

## Weights & Biases (wandb) guidance

- This project uses Weights & Biases to log experiment metrics, model checkpoints and artifacts. Recommended workflow:
	1. Create a WandB account and `wandb login` locally.
	2. When training, ensure the script is configured to use your WandB project and run name.
	3. Use WandB Artifacts to upload model weights and evaluation outputs rather than committing them to git:

If you need to add large model files and want them in the repo, use Git LFS:

```bash
# install Git LFS (macOS)
brew install git-lfs
git lfs install
git lfs track "Saved_model/*"
git add .gitattributes
git add Saved_model/your-large-model.bin
git commit -m "Add model via LFS"
git push
```

---

## Ethical notes & privacy

- To protect participant privacy, this repository does not include any raw or identifiable data.
- De-identify data where possible and include only derived or aggregated statistics in public releases.
- Deployment safeguards: automated tools should include human-in-the-loop review, rate-limited alerts, and clear escalation paths to crisis services.
- Transparency: provide model cards or documentation describing performance, limitations, and intended use cases.

