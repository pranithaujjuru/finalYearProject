import streamlit as st
import asyncio
import json
import os
from datetime import datetime
from PIL import Image
import base64
from python_agents.nlp_agent import extract_clinical_data
from python_agents.vision_agent import analyze_mri
from python_agents.validator_agent import validate_prognosis
from python_agents.translator_agent import translate_to_patient_voice
from python_agents.schemas import FinalReport, PatientVisitSummary

# --- PAGE CONFIG ---
st.set_page_config(page_title="Sciatica Patient Assistant", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS (Patient-Centric visual empathy) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background-color: #F9FAFB;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Patient-Centric Chat Bubbles */
        .stChatMessage {
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            border: 1px solid #F3F4F6;
            background-color: #FFFFFF !important;
            color: #1F2937 !important;
        }

        /* Ensure markdown inside chat messages is also dark */
        .stChatMessage div.stMarkdown p {
            color: #1F2937 !important;
        }
        
        /* Metric Styling */
        [data-testid="stMetricValue"] {
            font-size: 1.6rem;
            color: #111827;
            font-weight: 700;
        }
        
        /* Care Team Section Card */
        .care-card {
            background-color: #FFFFFF;
            padding: 28px;
            border-radius: 20px;
            border: 1px solid #E5E7EB;
            margin-bottom: 28px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.04);
        }
        
        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #1F2937;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .jargon-buster {
            background-color: #F0FDF4;
            border: 1px solid #DCFCE7;
            border-radius: 12px;
            padding: 12px;
            margin-top: 8px;
        }
        
        /* Amber Warning for Red Flags */
        .stAlert {
            border-radius: 12px !important;
            border: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_stage" not in st.session_state:
    st.session_state.chat_stage = "demographics"
if "patient_data" not in st.session_state:
    st.session_state.patient_data = {
        "age": None,
        "gender": None,
        "weight": None,
        "history": "",
        "mri_bytes": None,
        "mri_name": None,
        "mri_type": None
    }
if "prognosis_report" not in st.session_state:
    st.session_state.prognosis_report = None

# --- SIDEBAR (Patient Overview) ---
with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #111827; margin-bottom: 0;'>Patient File</h2>", unsafe_allow_html=True)
    st.divider()
    
    age_display = st.session_state.patient_data["age"] if st.session_state.patient_data["age"] else "..."
    gender_display = st.session_state.patient_data["gender"] if st.session_state.patient_data["gender"] else "..."
    weight_display = st.session_state.patient_data["weight"] if st.session_state.patient_data["weight"] else "..."
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Age", f"{age_display} yrs" if age_display != "..." else age_display)
        st.metric("Weight", weight_display)
    with c2:
        st.metric("Gender", gender_display)
    
    st.divider()
    if st.button("🗑️ Reset Consultation", use_container_width=True):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# --- HEADER ---
st.title("🛡️ Sciatica Care Assistant")
st.markdown("<p style='color: #6B7280; font-size: 1.1rem;'>Empathetic support for your recovery journey.</p>", unsafe_allow_html=True)

# Handle Chat View
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# --- CONVERSATION LOGIC ---
if not st.session_state.messages:
    welcome = """Hello. I am your **AI Care Team**. I’m here to help you understand what's happening and guide you through the next steps for your back health.
    
**To get started, could you please tell me your age, gender, and weight?**"""
    st.session_state.messages.append({"role": "assistant", "content": welcome})
    st.rerun()

# Handle Stage 0: Demographics
if st.session_state.chat_stage == "demographics":
    if prompt := st.chat_input("e.g. 45, Male, 80kg"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        tokens = prompt.replace(",", " ").split()
        for t in tokens:
            if t.isdigit() and not st.session_state.patient_data["age"]: 
                st.session_state.patient_data["age"] = t
            elif t.lower() in ["male", "female", "m", "f"]: 
                st.session_state.patient_data["gender"] = t.capitalize()
            elif "kg" in t.lower() or "lb" in t.lower():
                st.session_state.patient_data["weight"] = t
        
        res = "Thank you. Now, in your own words, please describe **where it hurts** and how long you've been feeling this way."
        st.session_state.messages.append({"role": "assistant", "content": res})
        st.session_state.chat_stage = "history"
        st.rerun()

# Handle Stage 1: History
elif st.session_state.chat_stage == "history":
    if prompt := st.chat_input("Describe your pain..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.patient_data["history"] = prompt
        
        res = "I understand. The final piece of information we need is your **Spinal MRI scan**. Please upload it below so the team can review the images."
        st.session_state.messages.append({"role": "assistant", "content": res})
        st.session_state.chat_stage = "mri_upload"
        st.rerun()

# Handle Stage 2: MRI Upload
elif st.session_state.chat_stage == "mri_upload":
    with st.chat_message("assistant"):
        st.markdown("**Awaiting your MRI scan image...**")
        uploaded = st.file_uploader("Upload MRI", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if uploaded:
            st.session_state.patient_data["mri_bytes"] = uploaded.getvalue()
            st.session_state.patient_data["mri_type"] = uploaded.type
            if st.button("✨ Complete Review", use_container_width=True):
                st.session_state.chat_stage = "processing"
                st.rerun()

# Handle Stage 3: Processing
elif st.session_state.chat_stage == "processing":
    with st.chat_message("assistant"):
        progress_text = st.empty()
        status_bar = st.progress(0)
        
        async def execute_care_team_review():
            # 1. Parallel Analysis
            progress_text.markdown("🤝 **The Care Team is starting the review...**")
            status_bar.progress(10)
            
            progress_text.markdown("📸 **AI Care Team is analyzing your MRI scan and clinical history...**")
            status_bar.progress(30)
            
            vision_task = analyze_mri(st.session_state.patient_data["mri_bytes"], st.session_state.patient_data["mri_type"])
            nlp_task = extract_clinical_data(st.session_state.patient_data["history"])
            
            # Run parallel
            vision, nlp = await asyncio.gather(vision_task, nlp_task)
            
            # 2. Safety Validation
            progress_text.markdown("🔬 **Comparing your scan with safety guidelines...**")
            status_bar.progress(60)
            valid = await validate_prognosis(nlp, vision.finding)
            
            # 3. Translation
            progress_text.markdown("✍️ **Creating your patient-centric summary...**")
            status_bar.progress(85)
            
            report = FinalReport(
                visionFindings=vision,
                clinicalData=nlp,
                validation=valid
            )
            
            summary = await translate_to_patient_voice(report)
            report.patientSummary = summary
            
            status_bar.progress(100)
            progress_text.markdown("✅ **Consultation Review Complete.**")
            return report

        try:
            # Use asyncio.run to manage the loop for the entire pipeline
            st.session_state.prognosis_report = asyncio.run(execute_care_team_review())
            st.session_state.chat_stage = "results"
            st.rerun()
        except Exception as e:
            st.error(f"Something went wrong during the review: {str(e)}")
            st.info("Technical Note: This can happen if the local AI models are busy or there is a memory conflict. Please ensure Ollama is running.")
            if st.button("Try again"): st.rerun()

# --- PATIENT-CENTRIC RESULTS RENDERING ---
if st.session_state.chat_stage == "results" and st.session_state.prognosis_report:
    r = st.session_state.prognosis_report
    s = r.patientSummary
    
    st.divider()
    st.markdown(f"## {s.summaryTitle}")
    st.markdown("Based on a complete review of your symptoms and your MRI scan, here is what is happening in your spine:")

    # Helper for Jargon Buster
    def jargon_buster(text):
        for item in s.jargonBuster:
            term = item['term']
            expl = item['explanation']
            if term.lower() in text.lower():
                with st.expander(f"💡 What does '{term}' actually mean?"):
                    st.info(expl)

    # 1. Primary Diagnosis & Imaging Analogy
    with st.container():
        st.markdown("<div class='care-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>🔍 Understanding the Cause</div>", unsafe_allow_html=True)
        st.markdown(s.diagnosis)
        st.markdown(f"**Imaging Insight:** {s.imaging}")
        jargon_buster(s.diagnosis + s.imaging)
        st.markdown("</div>", unsafe_allow_html=True)

    # 2. Neurological explanation
    with st.container():
        st.markdown("<div class='care-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>⚡ Your Nerve Function</div>", unsafe_allow_html=True)
        st.markdown(s.neurological)
        jargon_buster(s.neurological)
        st.markdown("</div>", unsafe_allow_html=True)

    # 3. Plan & Path Forward
    with st.container():
        st.markdown("<div class='care-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-title'>🌱 The Path Forward</div>", unsafe_allow_html=True)
        st.markdown(s.plan)
        st.markdown("</div>", unsafe_allow_html=True)

    # 4. Safety & Red Flags (Soft Amber)
    if r.validation.risks or s.redFlags:
        st.warning(f"**Emergency Awareness**\n\n{s.redFlags}")
        with st.expander("Why are these considered emergencies?"):
            st.markdown("These symptoms can sometimes signal that the nerves in your lower back are under significant stress that requires immediate medical attention to prevent long-term damage.")

    # Reset Button
    if st.button("Start New Review", use_container_width=True):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
