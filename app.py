import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import openai
import gdown

# === Download model if not exists ===
@st.cache_resource
def download_model():
    model_path = "enhanced_brain_tumor_classifier.h5"
    file_id = "1MvsNwDuDvkceJG7OXsJW7VQGE2se3lFa"
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(model_path):
        with st.spinner("Downloading model..."):
            gdown.download(url, model_path, quiet=False)
    return model_path

# === Load the model ===
@st.cache_resource
def load_model():
    model_path = download_model()
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# === OpenAI API Key from environment variable ===
openai.api_key = os.getenv("OPENAI_API_KEY")

# === App Title ===
st.set_page_config(page_title="Brain Tumor Detector", layout="centered")
st.title("🧠 Brain Tumor Detection Chatbot")
st.markdown("Upload an MRI image and answer the questions below to get a preliminary diagnostic report.")

# === Image Upload ===
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

# === Chatbot Questions ===
st.header("💬 Medical Chatbot Consultation")
symptoms = {}
questions = [
    ("headache", "Is the patient experiencing persistent headaches?"),
    ("seizures", "Has the patient experienced seizures?"),
    ("vision", "Any changes in vision or blurred vision?"),
    ("hearing", "Any hearing loss or ringing in the ears?"),
    ("balance", "Any issues with balance or coordination?"),
    ("nausea", "Any frequent nausea or vomiting?")
]

if "step" not in st.session_state:
    st.session_state.step = 0

if st.session_state.step < len(questions):
    key, question = questions[st.session_state.step]
    answer = st.radio(question, ("Yes", "No"), key=key)
    symptoms[key] = answer
    if st.button("Next"):
        st.session_state.step += 1
else:
    st.success("✅ All symptoms collected. You can now analyze the image.")

# === Prediction ===
if uploaded_file and st.session_state.step >= len(questions):
    if st.button("🔍 Analyze MRI"):
        try:
            img = Image.open(uploaded_file).convert("RGB")
            img = img.resize((299, 299))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]
            confidence = float(np.max(prediction))

            # === Show Result ===
            st.success(f"🧠 **Prediction**: `{predicted_class.upper()}` with confidence `{confidence:.2f}`")
            st.image(img, caption=f"Prediction: {predicted_class} ({confidence:.2f})", use_column_width=True)

            # === Report ===
            report = f"""
🧠 **Tumor Type**: {predicted_class.capitalize()}
📊 **Confidence**: {confidence:.2f}

**🩺 Reported Symptoms:**
- Headache: {symptoms.get('headache', 'N/A')}
- Seizures: {symptoms.get('seizures', 'N/A')}
- Vision issues: {symptoms.get('vision', 'N/A')}
- Hearing issues: {symptoms.get('hearing', 'N/A')}
- Balance issues: {symptoms.get('balance', 'N/A')}
- Nausea: {symptoms.get('nausea', 'N/A')}

🧠 **AI Remark**: This appears to be a case of **{predicted_class.upper()}**. Further medical examination is advised.
"""

            st.header("📝 Preliminary Report")
            st.markdown(report)

            # === GPT Medical Recommendation ===
            if openai.api_key:
                st.header("🤖 AI Medical Recommendation")
                gpt_prompt = f"Given the symptoms: {symptoms} and the AI prediction: {predicted_class}, provide a medical recommendation."
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=gpt_prompt,
                    max_tokens=150
                )
                st.write(response.choices[0].text.strip())
            else:
                st.warning("OpenAI API key not set. Set the `OPENAI_API_KEY` environment variable to enable GPT recommendations.")

            # === Download Button ===
            st.download_button("📄 Download Report", report, file_name="brain_tumor_report.txt")

        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
else:
    st.info("Please upload an MRI image and complete all questions.")
