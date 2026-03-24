import json
import re
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from python_agents.schemas import PatientVisitSummary, FinalReport
from python_agents.utils import extract_json

async def translate_to_patient_voice(report: FinalReport) -> PatientVisitSummary:
    """
    Translates technical clinical data into an empathetic, patient-centric summary
    using analogies and simple language.
    """
    llm = ChatOllama(model="llama3.2", temperature=0.3)
    parser = PydanticOutputParser(pydantic_object=PatientVisitSummary)

    template = """
    You are an empathetic, world-class doctor. Your job is to take structured clinical data and weave it into a single, conversational "Patient Visit Summary".
    
    INSTRUCTIONS:
    1. CONVERSATIONAL TONE: Speak directly to the patient ("Your scan shows...", "We noticed...").
    2. ANALOGIES: Translate all medical jargon into simple analogies. 
       - e.g., 'Thecal sac compression' -> 'pressure on the main nerve bundle, acting like a pinched hose'.
       - e.g., 'Foraminal stenosis' -> 'narrowing of the exit tunnels for your nerves'.
    3. NO JARGON IN MAIN TEXT: If you must use a term like 'L4-L5', explain its location (e.g., 'your lower back near the belt line').
    4. JARGON BUSTER: Provide a list of complex terms used in the original report and their simple 1-sentence explanations.
    5. SAFETY: Explain 'Red Flags' calmly but clearly as reasons to seek immediate help.
    
    DATA TO TRANSLATE:
    - Vision Findings: {vision}
    - Clinical Extraction: {clinical}
    - Validation/Safety: {validation}
    
    {format_instructions}
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["vision", "clinical", "validation"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    # Prepare input strings
    vision_str = f"Finding: {report.visionFindings.finding}, Severity: {report.visionFindings.severity}"
    clinical_str = f"Symptoms: {', '.join(report.clinicalData.symptoms)}, History: {report.clinicalData.history}"
    validation_str = f"Recommendation: {report.validation.recommendation}, Risks: {', '.join(report.validation.risks)}"

    chain = prompt | llm
    
    try:
        response = await chain.ainvoke({
            "vision": vision_str,
            "clinical": clinical_str,
            "validation": validation_str
        })
        
        data = extract_json(response.content)
        return PatientVisitSummary(**data)
    except Exception as e:
        print(f"Translator Agent Error: {e}")
        # Fallback summary
        return PatientVisitSummary(
            summaryTitle="Your Clinical Update",
            diagnosis="We are reviewing your back pain symptoms and scan.",
            neurological="Checking nerve function in your legs.",
            imaging="Analyzing the images of your spine.",
            plan="Continuing with the current investigation.",
            redFlags="Seek immediate care if you lose bowel/bladder control.",
            jargonBuster=[{"term": "Sciatica", "explanation": "Pain that travels along the large nerve from your back down your leg."}]
        )
