import streamlit as st
import joblib
from deep_translator import GoogleTranslator

# Load pipeline (TF-IDF + classifier)
@st.cache_resource
def load_model():
    return joblib.load('model_pipeline.pkl')

model = load_model()

# Translation helper
def translate_text(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text

st.title("Disaster Message Classifier with Creole–English Translation")

user_text = st.text_area("Enter a message (Creole or English):")

if st.button("Analyze"):
    if user_text.strip():
        translated = translate_text(user_text)
        st.write("**Translated:**", translated)

        # Predict
        pred = model.predict([translated])[0]
        prob = model.predict_proba([translated])[0].max()

        st.write("**Prediction:**", "🚨 Urgent" if str(pred).lower() =="request" else "✅ Not Urgent")
        st.write(f"**Confidence:** {prob*100:.2f}%")

        # Access TF-IDF inside pipeline
        vectorizer = model.named_steps['tfidf']
        X = vectorizer.transform([translated])
        dense = X.todense().tolist()[0]
        feature_names = vectorizer.get_feature_names_out()
        top_features = sorted(zip(feature_names, dense), key=lambda x: x[1], reverse=True)[:10]

        st.subheader("Top contributing words")
        for word, weight in top_features:
            st.write(f"{word}: {weight:.4f}")
    else:
        st.warning("Please enter a message first.")