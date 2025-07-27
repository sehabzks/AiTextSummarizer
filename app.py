from flask import Flask, render_template, request
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

# Modeli yükle
model_path = "model"  # localde ise tam yolu ver
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")

# Özetleme fonksiyonu
def generate_summary(text):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", truncation=True, padding="max_length", max_length=768)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Ana sayfa
@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    if request.method == "POST":
        input_text = request.form["input_text"]
        summary = generate_summary(input_text)
    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
