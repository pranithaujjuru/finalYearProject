import json
import re
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from core.schemas import PatientVisitSummary, FinalReport
from core.utils import extract_json

async def translate_to_patient_voice(report: FinalReport) -> PatientVisitSummary:
    """
    Translates technical clinical data into an empathetic, patient-centric summary
    using analogies and simple language.
    """
    llm = ChatOllama(model="llama3.2", temperature=0.3)
    parser = PydanticOutputParser(pydantic_object=PatientVisitSummary)
    # Prepare input strings
    vision_str = f"Finding: {report.visionFindings.finding}, Severity: {report.visionFindings.severity}"
    clinical_str = f"Symptoms: {', '.join(report.clinicalData.symptoms)}, History: {report.clinicalData.history}"
    validation_str = f"Recommendation: {report.validation.recommendation}, Risks: {', '.join(report.validation.risks)}"

    template = """
        [URGENT: DO NOT OUTPUT THE JSON SCHEMA. OUTPUT ONLY DATA.]
        You are an empathetic doctor explaining results to a patient.
        
        DATA:
        - Scan Findings: {vision}
        - Clinical Details: {clinical}
        - Safety Check: {validation}
        
        Create a warm, simple summary using analogies.
        Return ONLY a JSON object with:
        - summaryTitle: (empathetic title)
        - diagnosis: (explanation of condition)
        - neurological: (explanation of nerve findings)
        - imaging: (simple translation of MRI)
        - plan: (next steps in care)
        - redFlags: (emergency warnings)
        - jargonBuster: (list of {{"term": "...", "explanation": "..."}} pairs)

        JSON:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["vision", "clinical", "validation"]
    )

    chain = prompt | llm
    
    try:
        response = await chain.ainvoke({
            "vision": vision_str,
            "clinical": clinical_str,
            "validation": validation_str
        })
        
        content = str(response.content)
        data = extract_json(content)
        
        # If the model returned a 'properties' wrapper (common schema leak)
        if isinstance(data, dict) and "properties" in data:
            props = data.get("properties", {})
            if isinstance(props, dict):
                data = {k: v.get("default", v.get("example", "")) if isinstance(v, dict) else v 
                        for k, v in props.items()}
        
        if not isinstance(data, dict):
            raise ValueError("Parsed data is not a dictionary")

        return PatientVisitSummary(**data)

    except Exception as e:
        print(f"Translator Agent Error: {e}")
        # Dynamic Fallback using actual report data (Unified & Informative)
        findings_summary = report.visionFindings.finding if report.visionFindings.finding else "Spinal features analyzed."
        symptoms_str = ", ".join(report.clinicalData.symptoms) if report.clinicalData.symptoms else "back-related symptoms"
        
        return PatientVisitSummary(
            summaryTitle="Clinical Insights & Recovery Plan",
            diagnosis=f"Based on your report, we are focusing on a finding described as: {findings_summary}. This relates to your reported symptoms of {symptoms_str}.",
            neurological=f"Our AI Care Team is monitoring your nerve function closely. Your history of {report.clinicalData.history[:100]}... provides important context for this care.",
            imaging=f"The MRI scan detected '{findings_summary}' at the {report.visionFindings.location or 'monitored'} level of your spine.",
            plan="We recommend a professional clinical review of these automated findings to coordinate the best physical therapy or specialist path for you.",
            redFlags="⚠️ Seek immediate medical attention if you experience sudden weakness, numbness in the saddle area, or loss of bowel/bladder control.",
            jargonBuster=[{"term": "Herniation", "explanation": "A common condition where a spinal disc bulges, potentially touching a nerve."}]
        )
