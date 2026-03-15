import streamlit as st
import prediction
from PIL import Image
import numpy as np
import os

# Setup the page 
st.set_page_config(page_title="Chest X-Ray Analyzer (Scikit-Learn)", page_icon="🫁", layout="wide")

st.title("🫁 Chest X-Ray Analyzer")
st.markdown("### Powered by Scikit-Learn Random Forest")
st.info("This version uses a `.pkl` model to ensure maximum stability and speed on Streamlit Cloud.")

uploaded_file = st.file_uploader("Upload a Chest X-Ray Image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded X-Ray", use_column_width=True)
        
    with col2:
        with st.spinner("Analyzing image..."):
            try:
                # Save temp file
                with open("temp.png", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Get prediction
                probs = prediction.get_prediction("temp.png")
                classes = ['COVID', 'Normal', 'Viral Pneumonia']
                
                # Show results
                st.subheader("Analysis Results")
                
                # Find best class
                best_idx = np.argmax(probs)
                confidence = probs[best_idx] * 100
                
                st.metric("Top Prediction", classes[best_idx], f"{confidence:.1f}% Confidence")
                
                # Show bars
                for cls, prob in zip(classes, probs):
                    st.write(f"**{cls}**")
                    st.progress(float(prob))
                    
            except Exception as e:
                st.error(f"Error during analysis: {e}")
            finally:
                if os.path.exists("temp.png"):
                    os.remove("temp.png")
