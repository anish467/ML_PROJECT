import streamlit as st
import requests
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import re

# Function to extract ASIN
def extract_asin(url):
    match = re.search(r'/dp/([A-Z0-9]{10})|/gp/product/([A-Z0-9]{10})|/product-reviews/([A-Z0-9]{10})', url)
    if match:
        return next(g for g in match.groups() if g)
    return None

# Load tokenizer and model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        tokenizer, model = pickle.load(f)
    return tokenizer, model

# Streamlit UI
st.title("🛍️ Amazon Review Authenticity Checker")
st.write("Enter an Amazon product URL (India) to analyze the authenticity of its reviews.")

# Input
url = st.text_input("🔗 Product URL", placeholder="https://www.amazon.in/...")
submit = st.button("Analyze Reviews")

if submit and url:
    asin = extract_asin(url)
    
    if not asin:
        st.error("❌ Invalid Amazon URL or ASIN not found.")
    else:
        st.info(f"📦 Extracted ASIN: `{asin}`")
        
        tokenizer, model = load_model()
        
        # Fetch reviews from Oxylabs
        USERNAME = 'anany_HR0gy'
        PASSWORD = 'Anany_10102004'

        payload = {
            'source': 'amazon_reviews',
            'domain': 'in',
            'query': asin,
            'pages': 10,
            'parse': True,
        }

        with st.spinner("Fetching and analyzing reviews..."):
            response = requests.post(
                'https://realtime.oxylabs.io/v1/queries',
                auth=(USERNAME, PASSWORD),
                json=payload
            )

            if response.status_code != 200:
                st.error(f"❌ Failed to fetch reviews: {response.status_code}")
            else:
                data = response.json()
                review_texts = []
                for result in data.get("results", []):
                    reviews = result.get("content", {}).get("reviews", [])
                    review_texts.extend([review.get("content", '') for review in reviews if 'content' in review])

                if not review_texts:
                    st.warning("⚠️ No reviews found for this product.")
                else:
                    sequences = tokenizer.texts_to_sequences(review_texts)
                    padded = pad_sequences(sequences, maxlen=100, padding='post')
                    predictions = model.predict(padded)
                    binary_preds = (predictions > 0.5).astype(int).flatten()

                    total = len(binary_preds)
                    original = int(np.sum(binary_preds))
                    fake = total - original
                    percentage = (original / total) * 100

                    st.success(f"✅ {percentage:.2f}% of the reviews are predicted to be **ORIGINAL**.")
                    st.write(f"🟢 Original: {original}")
                    st.write(f"🔴 Fake: {fake}")

                    # Plot
                    fig, ax = plt.subplots()
                    ax.bar(['Original', 'Fake'], [original, fake], color=['green', 'red'])
                    ax.set_title('Review Authenticity Prediction')
                    ax.set_ylabel('Number of Reviews')
                    st.pyplot(fig)
