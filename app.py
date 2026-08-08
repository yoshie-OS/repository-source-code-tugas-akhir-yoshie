import torch
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label2id = {'anger': 0, 'fear': 1, 'joy': 2, 'love': 3, 'sadness': 4}
id2label = {v: k for k, v in label2id.items()}
COLORS = {
    'anger': '#EF5350',
    'fear': '#AB47BC',
    'joy': '#FFA726',
    'love': '#EC407A',
    'sadness': '#42A5F5'
}

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
        text, max_length=128, padding='max_length',
        truncation=True, return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        pred_id = probs.argmax()

    return id2label[pred_id], probs

def plot_probs(probs, title):
    labels = list(id2label.values())
    colors = [COLORS[l] for l in labels]
    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.barh(labels, probs, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Probability')
    ax.set_title(title)
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%', va='center', fontsize=9)
    plt.tight_layout()
    return fig

# session state untuk agreement tracking
if 'agreement_count' not in st.session_state:
    st.session_state.agreement_count = 0
if 'total_count' not in st.session_state:
    st.session_state.total_count = 0

st.title("Klasifikasi Emosi Tweet Film Piala Citra 2025")
st.caption("Evaluasi Lintas Domain — IndoBERT")

tokenizer, model_cross, model_baseline = load_models()

text = st.text_area("Masukkan tweet:", placeholder="Contoh: film ini bagus banget!")

if st.button("Prediksi") and text:
    label_cross, probs_cross = predict(text, model_cross, tokenizer)
    label_baseline, probs_baseline = predict(text, model_baseline, tokenizer)

    # update agreement tracking
    st.session_state.total_count += 1
    if label_cross == label_baseline:
        st.session_state.agreement_count += 1

    # highlight kalau beda
    if label_cross != label_baseline:
        st.warning(f"⚠️ Dua model tidak sepakat — Cross-Domain: **{label_cross.upper()}**, Baseline: **{label_baseline.upper()}**")
    else:
        st.success(f"✅ Kedua model sepakat: **{label_cross.upper()}**")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cross-Domain Model")
        st.markdown(f"Prediksi: **:{COLORS[label_cross].replace('#','')}[{label_cross.upper()}]**")
        fig = plot_probs(probs_cross, "Confidence Score")
        st.pyplot(fig)

    with col2:
        st.subheader("In-Domain Baseline")
        st.markdown(f"Prediksi: **{label_baseline.upper()}**")
        fig = plot_probs(probs_baseline, "Confidence Score")
        st.pyplot(fig)

    # agreement rate
    if st.session_state.total_count > 0:
        rate = st.session_state.agreement_count / st.session_state.total_count
        st.metric(
            label="Agreement Rate (sesi ini)",
            value=f"{rate*100:.1f}%",
            delta=f"{st.session_state.agreement_count}/{st.session_state.total_count} prediksi sepakat"
        )