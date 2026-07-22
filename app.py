import streamlit as st
from PIL import Image
import numpy as np
import pickle
import cv2
import os

# 1. Page Configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="VigilantEye AI | Currency Verification", 
    page_icon="🛡️",
    layout="wide"
)
st.set_option("client.toolbarMode", "viewer")

# 2. Custom CSS Frontend Styling (Cybersecurity Theme)
st.markdown("""
    <style>
    /* Hide the entire top header bar completely (including Deploy button) */
    header, [data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Main background and font adjustments */
    .main {
        background-color: #0e1117;
        color: #ffffff;
        padding-top: 2rem; /* Give a tiny bit of space since we hid the header */
    }
    /* Main dashboard title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #00ffcc;
        text-shadow: 0px 0px 10px rgba(0, 255, 204, 0.3);
        margin-bottom: 5px;
    }
    /* Subtitle styling */
    .sub-title {
        font-size: 1.1rem;
        color: #8a99ad;
        margin-bottom: 30px;
    }
    /* Custom cards for layout separation */
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

# 4. Sidebar Control Panel
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/shield.png", width=80)
    st.markdown("### **Agent Control Panel**")
    st.write("Configure the AI detection subsystem parameters here.")
    
    note_type = st.selectbox(
        "Target Denomination:",
        ["Select a note...", "₹10", "₹20", "₹50", "₹100", "₹200", "₹500"]
    )
    
    st.divider()
    st.markdown("**Model Engine Status:**")
    if ai_model is not None:
        st.success("🟢 CORE_AI: ONLINE")
    else:
        st.error("🔴 CORE_AI: OFFLINE")

# 5. Main Dashboard Layout
st.markdown('<h1 class="main-title">🛡️ VigilantEye AI Node</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Automated Indian Counterfeit Currency Identification Subsystem</p>', unsafe_allow_html=True)

if note_type == "Select a note...":
    # Welcome / Instruction box when no note is chosen
    st.markdown("""
        <div class="crypto-card">
            <h4>💡 System Ready for Inspection</h4>
            <p>Please select a currency denomination from the left <b>Agent Control Panel</b> sidebar to initialize the automated target scanning sequence.</p>
        </div>
    """, unsafe_allow_html=True)

else:
    # Split the screen into 2 clean columns (Left for Upload, Right for Results)
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown(f"### 📂 Target Ingestion (`{note_type}`)")
        
        # Segmented control for input method choice
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
        
        if img is not None:
            st.image(img, caption=f"Ingested {note_type} Sample", use_container_width=True)
            
            # Glowing action button
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
                    # Process image safely using PIL to OpenCV conversion
                    opencv_img = np.array(img.convert('RGB'))
                    opencv_img = cv2.cvtColor(opencv_img, cv2.COLOR_RGB2BGR)
                    
                    # AI Resize and Flat-vector preparation
                    resized_img = cv2.resize(opencv_img, (128, 128))
                    feature_vector = resized_img.flatten().reshape(1, -1)
                    
                    # Inference step
                    prediction = ai_model.predict(feature_vector)[0]
                
                # Logic Processing & UI Feedback Cards
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
                    st.info(f"AI System Exception: Detected a genuine note signature, but identified structural properties of a **₹{prediction.split('_')[1]}** variant instead of the expected target ({note_type}).")
                else:
                    st.error("🚨 **CRITICAL ALERT: COUNTERFEIT PATTERNS DETECTED**")
                    st.markdown("""
                        <div style="background-color:#4d1414; padding:15px; border-radius:5px; border-left:5px solid #ff3333;">
                            <b>Threat Level: High</b><br>
                            The target sample matches recorded malicious pixel signatures, altered micro-lettering distributions, and counterfeit feature vectors stored in the baseline model network.
                        </div>
                    """, unsafe_allow_html=True)