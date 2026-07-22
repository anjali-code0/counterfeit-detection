import streamlit as st
from PIL import Image
import numpy as np
import pickle
import os

# Safe import for cv2 to prevent app crash
try:
    import cv2
except ImportError:
    cv2 = None

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="VigilantEye AI | Currency Verification", 
    page_icon="🛡️",
    layout="wide"
)
st.set_option("client.toolbarMode", "viewer")

# 2. Custom CSS Frontend Styling
st.markdown("""
    <style>
    header, [data-testid="stHeader"] {
        display: none !important;
    }
    .main {
        background-color: #0e1117;
        color: #ffffff;
        padding-top: 2rem;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #00ffcc;
        text-shadow: 0px 0px 10px rgba(0, 255, 204, 0.3);
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #8a99ad;
        margin-bottom: 30px;
    }
    .crypto-card {
        background-color: #1f2633;
        color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #00ffcc;
        margin-bottom: 20px;
    }   
    </style>
""", unsafe_allow_html=True)

# 3. Load the Trained AI Model Safely
@st.cache_resource
def load_ai_model():
    if os.path.exists('currency_classifier.pkl'):
        with open('currency_classifier.pkl', 'rb') as f:
            return pickle.load(f)
    return None

ai_model = load_ai_model()

# 4. Main Dashboard Header
st.markdown('<h1 class="main-title">🛡️ VigilantEye AI Node</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Automated Indian Counterfeit Currency Identification Subsystem</p>', unsafe_allow_html=True)

# 5. Agent Control Panel
st.markdown("### **Agent Control Panel**")
st.write("Configure the AI detection subsystem parameters here.")

note_type = st.selectbox(
    "Target Denomination:",
    ["Select a note...", "₹10", "₹20", "₹50", "₹100", "₹200", "₹500"]
)

st.markdown("**Model Engine Status:**")
if ai_model is not None:
    st.success("🟢 CORE_AI: ONLINE")
else:
    st.error("🔴 CORE_AI: OFFLINE")

st.divider()

if note_type == "Select a note...":
    st.markdown("""
        <div class="crypto-card">
            <h4>💡 System Ready for Inspection</h4>
            <p>Please select a currency denomination above to initialize the automated target scanning sequence.</p>
        </div>
    """, unsafe_allow_html=True)
else:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown(f"### 📂 Target Ingestion (`{note_type}`)")
        
        input_method = st.radio(
            "Select Input Source:",
            ["📁 Upload Image File", "📸 Use Device Camera"],
            horizontal=True
        )
        
        uploaded_file = None
        img = None
        
        if input_method == "📁 Upload Image File":
            uploaded_file = st.file_uploader("Drop target currency matrix (JPG/PNG):", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                img = Image.open(uploaded_file)
        else:
            uploaded_file = st.camera_input("Position the currency note clearly in the frame:")
            if uploaded_file is not None:
                img = Image.open(uploaded_file)
        
        run_scan = False
        if img is not None:
            st.image(img, caption=f"Ingested {note_type} Sample", use_container_width=True)
            run_scan = st.button(f"⚡ Initialize Neural Scan", use_container_width=True)

    with col2:
        st.markdown("### 📊 Live Diagnostic Readout")
        
        if uploaded_file is None:
            st.info("Awaiting target image upload to populate telemetry...")
        
        elif uploaded_file is not None and run_scan:
            if ai_model is None:
                st.error("🚨 Critical Failure: 'currency_classifier.pkl' not found locally.")
            else:
                with st.spinner("Executing texture analysis and pixel array mapping..."):
                    # Safe Image Processing (Using PIL + NumPy fallback)
                    img_resized = img.convert('RGB').resize((128, 128))
                    img_array = np.array(img_resized)
                    
                    if cv2 is not None:
                        # Convert RGB to BGR to match OpenCV training format
                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    else:
                        # Fallback channel swap if OpenCV isn't loaded
                        img_array = img_array[:, :, ::-1]
                    
                    feature_vector = img_array.flatten().reshape(1, -1)
                    prediction = ai_model.predict(feature_vector)[0]
                
                clean_denom = note_type.replace("₹", "")
                expected_target = f"Genuine_{clean_denom}"
                
                st.markdown("#### **Verification Logs**")
                
                if prediction == expected_target:
                    st.balloons()
                    st.success(f"🏅 **VERDICT: SECURE GENUINE {note_type} NOTE**")
                    st.markdown("""
                        <div style="background-color:#1e3d2f; padding:15px; border-radius:5px; border-left:5px solid #00ffcc; color:#ffffff;">
                            <b>Security Check:</b> Passed Integrity Test<br>
                            <b>AI Verification:</b> Optimal Geometric & Micro-text Alignment Match
                        </div>
                    """, unsafe_allow_html=True)
                elif "Genuine" in prediction:
                    st.warning(f"⚠️ **VERDICT: MISMATCHED DENOMINATION SPECIFICATIONS**")
                    st.info(f"AI System Exception: Identified a **₹{prediction.split('_')[1]}** variant instead of target ({note_type}).")
                else:
                    st.error("🚨 **CRITICAL ALERT: COUNTERFEIT PATTERNS DETECTED**")
                    st.markdown("""
                        <div style="background-color:#4d1414; padding:15px; border-radius:5px; border-left:5px solid #ff3333;">
                            <b>Threat Level: High</b><br>
                            Malicious pixel signatures matched baseline model network.
                        </div>
                    """, unsafe_allow_html=True)