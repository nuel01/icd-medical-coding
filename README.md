# 🏥 ICD-10 Medical Coding with Fine-tuned LLMs

Automated ICD-10 code assignment from clinical text using fine-tuned large language models. This project fine-tunes both **Google MedGemma-4b** and **Llama3-OpenBioLLM-8B** on ICD medical coding tasks, with a live Gradio demo deployed on Hugging Face Spaces.

---

## 🔗 Resources

| Resource | Link |
|---|---|
| 🤗 MedGemma-4b-ICD Model | [huggingface.co/abnuel/MedGemma-4b-ICD](https://huggingface.co/abnuel/MedGemma-4b-ICD) |
| 🤗 OpenBioLLM-8B Model | [huggingface.co/abnuel/fine-tuned-openbiollm-medical-coding](https://huggingface.co/abnuel/fine-tuned-openbiollm-medical-coding) |
| 🚀 Live Demo | [huggingface.co/spaces/abnuel/med-coding](https://huggingface.co/spaces/abnuel/med-coding) |

---

## 📌 Overview

ICD (International Classification of Diseases) coding is a critical but labor-intensive clinical workflow. Human coders must read clinical notes and assign standardized ICD-10 codes used for billing, epidemiology, and care tracking. This project automates that process using instruction-tuned biomedical LLMs.

**Two models were fine-tuned and compared:**

- **MedGemma-4b-ICD** — Google's MedGemma (4B params), purpose-built for medical language understanding
- **fine-tuned-openbiollm-medical-coding** — Meta's Llama 3 base, fine-tuned by Saama AI on biomedical corpora, then further fine-tuned here for ICD coding

Both were trained using **Supervised Fine-Tuning (SFT)** via the TRL library.

---

## 🗂️ Project Structure

```
icd-medical-coding/
├── README.md
├── requirements.txt
├── notebooks/
│   └── icd_coding_finetune.ipynb   # End-to-end fine-tuning walkthrough
├── src/
│   ├── train.py                     # SFT training script
│   ├── inference.py                 # Inference utilities
│   └── evaluate.py                  # Evaluation metrics
└── demo/
    └── app.py                       # Gradio demo (mirrors HF Space)
```

---

## ⚙️ Setup

```bash
git clone https://github.com/nuel01/icd-medical-coding.git
cd icd-medical-coding
pip install -r requirements.txt
```

---

## 🚀 Quick Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "abnuel/MedGemma-4b-ICD"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = """You are a clinical coding assistant. Assign the correct ICD-10 code(s) 
for the following clinical note.

Clinical note: Patient presents with uncontrolled type 2 diabetes mellitus 
with diabetic peripheral neuropathy.

ICD-10 Code(s):"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.1, do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 🏋️ Training

Fine-tuning was performed using [TRL's SFT Trainer](https://huggingface.co/docs/trl/sft_trainer):

```python
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# See notebooks/icd_coding_finetune.ipynb for the full training pipeline
```

Key training decisions:
- **SFT** over RLHF due to the structured, deterministic nature of ICD coding
- Prompt formatting follows an instruction-response pattern to leverage the instruct-tuned base
- Hyperparameter tuning focused on output consistency and code format adherence

---

## 📊 Models Compared

| Model | Base | Params | Method |
|---|---|---|---|
| MedGemma-4b-ICD | google/medgemma-4b-it | 4B | SFT (TRL) |
| fine-tuned-openbiollm | aaditya/Llama3-OpenBioLLM-8B | 8B | SFT (TRL) |

---

## ⚠️ Limitations & Disclaimer

- Model outputs are intended for **research and decision support only**
- Should not replace certified medical coders for official billing or clinical records
- Performance varies across medical specialties and ICD-10 editions
- Not evaluated for ICD-10-PCS (procedural) codes

---

## 👤 Author

**Abayomi Adegunlehin** — AI/ML Engineer | Houston, TX  
[Hugging Face](https://huggingface.co/abnuel) · [GitHub](https://github.com/nuel01)

---

## 📄 License

This project is released under the MIT License. Model weights are subject to their respective base model licenses (Gemma Terms / Llama 3 Community License).
