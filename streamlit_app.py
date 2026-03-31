import streamlit as st
import asyncio
import json
import os
from datetime import datetime
from PIL import Image
import base64
from core.nlp_agent import extract_clinical_data
from core.vision_agent import analyze_mri
from core.validator_agent import validate_prognosis
from core.translator_agent import translate_to_patient_voice
from core.schemas import FinalReport, PatientVisitSummary

# --- PAGE CONFIG ---
st.set_page_config(page_title="Sciatica Patient Assistant", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS (Patient-Centric visual empathy) ---
st.markdown("""
    <style>
        /* High-Level Theme Awareness */
        .stApp {
            background-color: transparent !important;
        }

        /* Unified Card Styling with theme-aware colors */
        .care-card {
            background-color: var(--secondary-background-color);
            padding: 24px;
            border-radius: 16px;
            border: 1px solid rgba(128, 128, 128, 0.1);
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            color: var(--text-color);
        }
        
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-color);
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        /* Patient-Centric Chat Bubbles */
        .stChatMessage {
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            border: 1px solid rgba(128, 128, 128, 0.1);
            background-color: var(--secondary-background-color) !important;
            color: var(--text-color) !important;
        }

        .stChatMessage div.stMarkdown p {
            color: var(--text-color) !important;
        }
        
        .jargon-buster {
            background-color: rgba(240, 253, 244, 0.05);
            border: 1px solid rgba(220, 252, 231, 0.1);
            border-radius: 12px;
            padding: 12px;
            margin-top: 8px;
        }
        
        /* Correcting Alert Text Contrast (Yellow block issue) */
        .stAlert {
            border-radius: 12px !important;
            border: 1px solid rgba(128, 128, 128, 0.1) !important;
        }
        .stAlert div[role="alert"] p, .stAlert h3 {
            color: inherit !important;
            font-weight: 500;
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
    st.markdown(f"""
        <div style='text-align: center; border-bottom: 2px solid var(--medical-border); padding-bottom: 10px; margin-bottom: 20px;'>
            <h2 style='color: var(--text-color); margin: 0;'>📋 Patient File</h2>
        </div>
    """, unsafe_allow_html=True)
    
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
    
    # --- SCIATICA 101 ---
    st.markdown("### 🩺 About Sciatica")
    st.info("""
    The **Sciatic Nerve** is the largest nerve in your body. It runs from your lower back, through your hips, and down each leg. 
    
    When this nerve is compressed (often by a disc), it causes the radiating pain, numbness, or weakness known as Sciatica.
    """)
    
    # --- CARE GUIDELINES ---
    st.markdown("### 💡 Daily Care Tips")
    st.markdown("""
    - 🚶 **Keep Moving**: Gentle walking helps maintain flexibility.
    - 🪑 **Sit Smart**: Use a lumbar roll or small pillow for back support.
    - 🏋️ **Lift Safely**: Always bend your knees—never your back.
    - ❄️ **Cold/Heat**: Use ice for 15m to reduce swelling, then heat to relax muscles.
    """)
    
    st.divider()
    if st.button("🗑️ Reset Consultation", use_container_width=True):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# --- SYSTEM DIAGNOSTICS (Ensure Backend Health) ---
def run_system_health_check():
    """Checks if required models and Ollama are reachable."""
    health_results = []
    
    # 1. Check Ollama reachability
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            models = [m['name'] for m in resp.json().get('models', [])]
            required = ["llama3.2:latest", "llava:latest"]
            for r in required:
                if any(r in m for m in models):
                    health_results.append(f"✅ Model `{r}` is active.")
                else:
                    health_results.append(f"❌ Model `{r}` is missing. Run `ollama pull {r.split(':')[0]}`")
        else:
            health_results.append("❌ Ollama is unreachable (Status check failed).")
    except Exception:
        health_results.append("❌ Ollama is not running. Please start the Ollama application.")
        
    # 2. Check Embeddings Cache
    if os.path.exists("./models/embeddings"):
        health_results.append("✅ Local Embeddings cache is ready.")
    else:
        health_results.append("⚠️ Local Embeddings cache missing. First run may be slow.")
        
    return health_results

with st.sidebar:
    with st.expander("🩺 System Health", expanded=False):
        for status in run_system_health_check():
            st.write(status)

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
    
    # Helper for Jargon Buster
    def jargon_buster(text):
        if not s or not s.jargonBuster:
            return
        for item in s.jargonBuster:
            term = item['term']
            expl = item['explanation']
            if term.lower() in text.lower():
                with st.expander(f"💡 What does '{term}' actually mean?"):
                    st.info(expl)

    # --- NEW CONSOLIDATED DASHBOARD CARD ---
    st.markdown(f"""
<div class='care-card'>
<div style='font-size: 1.5rem; font-weight: 800; margin-bottom: 20px; border-bottom: 2px solid var(--medical-border); padding-bottom: 10px; color: var(--text-color);'>
🏥 Clinical Summary Dashboard
</div>
<div style='display: grid; grid-template-columns: 1fr; gap: 20px;'>
<!-- OVERALL SUMMARY -->
<div>
<div class='section-title'>📝 Overall Summary</div>
<div style='font-style: italic; opacity: 0.9;'>{s.summaryTitle}</div>
<div style='margin-top: 8px;'>Based on your symptoms and MRI, here is the focus of your care plan.</div>
</div>
<hr style='border: 0; border-top: 1px solid rgba(128,128,128,0.1); margin: 5px 0;'>
<!-- DETECTION & CAUSE -->
<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
<div>
<div class='section-title'>🔍 Detection (MRI)</div>
<div style='background: rgba(128,128,128,0.05); padding: 12px; border-radius: 8px;'>{s.imaging}</div>
</div>
<div>
<div class='section-title'>🩺 Probable Cause</div>
<div style='background: rgba(128,128,128,0.05); padding: 12px; border-radius: 8px;'>{s.diagnosis}</div>
</div>
</div>
<hr style='border: 0; border-top: 1px solid rgba(128,128,128,0.1); margin: 5px 0;'>
<!-- PRECAUTIONS & CONSULTANCY -->
<div style='display: grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
<div>
<div class='section-title'>⚠️ Precautions</div>
<div style='color: #ef4444; font-weight: 600;'>Emergency Red Flags:</div>
<div style='font-size: 0.9rem;'>{s.redFlags}</div>
</div>
<div>
<div class='section-title'>👨‍⚕️ Doctor's Consultancy</div>
<div style='border-left: 4px solid #3b82f6; padding-left: 12px; font-weight: 500;'>{r.validation.recommendation}</div>
</div>
</div>
<hr style='border: 0; border-top: 1px solid rgba(128,128,128,0.1); margin: 5px 0;'>
<!-- NERVE FUNCTION & PLAN -->
<div>
<div class='section-title'>⚡ Nerve Function & Recovery Plan</div>
<div>{s.neurological}</div>
<div style='margin-top: 10px; font-weight: 600; color: #10b981;'>Next Steps:</div>
<div>{s.plan}</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    # 5. TECHNICAL VALIDATION TABLE (Transparency)
    st.markdown("### 🛠️ Clinical & Technical Validation")
    validation_data = {
        "Metric": ["Confidence Score", "Safety Assessment", "Primary Recommendation"],
        "Value": [
            f"{r.validation.confidenceScore}%",
            "✅ Pass (Safe)" if r.validation.isSafe else "⚠️ Review Required",
            r.validation.recommendation
        ]
    }
    st.table(validation_data)

    # Jargon Buster (Move to bottom for cleanliness)
    jargon_buster(s.diagnosis + s.imaging + s.neurological)

    st.markdown("#### 📚 Scientific References & Guidelines")
    
    # Professional Clinical Guideline Mapping
    GUIDELINE_LINKS = {
        "NICE Guideline [NG59]": "https://www.nice.org.uk/guidance/ng59",
        "NHS Cauda Equina Standards": "https://www.nhs.uk/conditions/cauda-equina-syndrome/",
        "Spine-health Sciatica Guide": "https://www.spine-health.com/conditions/sciatica/what-you-need-know-about-sciatica",
        "AANS Herniated Disc Overview": "https://www.aans.org/patients/conditions-treatments/herniated-disc/",
        "RAG Guidelines": "https://www.nice.org.uk/guidance/ng59" # Alias for NICE
    }

    if r.validation.referencedGuidelines:
        for g_name in r.validation.referencedGuidelines:
            # Map generic labels to professional ones
            display_name = "NICE Clinical Guideline [NG59]" if g_name == "RAG Guidelines" else g_name
            url = GUIDELINE_LINKS.get(g_name, "https://www.nice.org.uk/guidance/ng59")
            st.markdown(f"- [{display_name}]({url})")
    else:
        st.info("The AI Care Team has cross-referenced these findings with internal clinical guidelines.")

    # 6. Safety Alert Block (Redundant but visible for urgency)
    if s.redFlags:
        st.divider()
        st.warning(f"### ⚠️ Emergency Awareness\n\n{s.redFlags}")

    # Reset Button
    st.divider()
    if st.button("Start New Review", use_container_width=True):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
