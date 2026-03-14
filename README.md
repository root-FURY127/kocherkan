# PrivacyGuardian Chatbot Model

This model is fine‑tuned from Mistral‑7B‑Instruct to handle privacy requests under the DPDP Act. It outputs structured JSON commands for backend execution.

## Usage

```python
from transformers import pipeline

pipe = pipeline("text-generation", model="Abinrajvb/Kocherkan")
prompt = "User: delete my phone number\nAssistant:"
output = pipe(prompt, max_new_tokens=100)[0]["generated_text"]
print(output)
