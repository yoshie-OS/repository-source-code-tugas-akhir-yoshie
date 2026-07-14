import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label2id = {'anger': 0, 'fear': 1, 'joy': 2, 'love': 3, 'sadness': 4}
id2label = {v: k for k, v in label2id.items()}

@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    
    model_cross = AutoModelForSequenceClassification.from_pretrained(
        "indobenchmark/indobert-base-p1", num_labels=5, id2label=id2label, label2id=label2id
    )
    model_cross.load_state_dict(torch.load("final_model.pt", map_location=device))
    model_cross = model_cross.to(device)
    model_cross.eval()

    model_baseline = AutoModelForSequenceClassification.from_pretrained(
        "indobenchmark/indobert-base-p1", num_labels=5, id2label=id2label, label2id=label2id
    )
    model_baseline.load_state_dict(torch.load("baseline_model_final.pt", map_location=device))
    model_baseline = model_baseline.to(device)
    model_baseline.eval()

    return tokenizer, model_cross, model_baseline

def predict(text, model, tokenizer):
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred_id = probs.argmax()

    return id2label[pred_id], probs

st.title("Klasifikasi Emosi Tweet Film Piala Citra 2025")
st.caption("Evaluasi Lintas Domain — IndoBERT")

tokenizer, model_cross, model_baseline = load_models()

text = st.text_area("Masukkan tweet:", placeholder="Contoh: film ini bagus banget!")

if st.button("Prediksi") and text:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cross-Domain Model")
        label, probs = predict(text, model_cross, tokenizer)
        st.success(f"**{label.upper()}**")
        for i, (l, p) in enumerate(zip(id2label.values(), probs)):
            st.progress(float(p), text=f"{l}: {p*100:.2f}%")

    with col2:
        st.subheader("In-Domain Baseline")
        label, probs = predict(text, model_baseline, tokenizer)
        st.success(f"**{label.upper()}**")
        for i, (l, p) in enumerate(zip(id2label.values(), probs)):
            st.progress(float(p), text=f"{l}: {p*100:.2f}%")