import json
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from .schemas import ClinicalExtraction, PatientQuestionnaire, Demographics
from .utils import extract_json

def synthesize_patient_answers(answers: PatientQuestionnaire) -> str:
    return f"Patient reports pain trajectory as {answers.painTrajectory} lasting for {answers.painDuration}. Previous treatments include: {answers.previousTreatments}. Regarding red flags, patient states: {answers.redFlags}. Patient describes neurological symptoms as: {answers.neurologicalSymptoms}."

async def process_patient_intake(answers: PatientQuestionnaire) -> ClinicalExtraction:
    narrative_text = synthesize_patient_answers(answers)
    return await extract_clinical_data(narrative_text)

async def extract_clinical_data(report_text: str) -> ClinicalExtraction:
    """
    Extracts clinical data using a local Llama3.2 model with aggressive schema stripping.
    """
    try:
        model = ChatOllama(model="llama3.2")
        parser = PydanticOutputParser(pydantic_object=ClinicalExtraction)
        
        template = """
        You are an expert Clinical NLP specialist. Extract precise data from the report below into valid JSON.
        
        GUIDELINES:
        - Symptoms: List specific medical terms (e.g., "L5 Radiculopathy", "Acute lower back pain").
        - Procedure: Suggest based on severity if not named:
            * "Physical Therapy" for mild/moderate pain.
            * "Epidural Steroid Injection" for persistent radiating pain.
            * "Emergency Surgery" for loss of bowel/bladder or severe weakness.
            * "MRI/Consultation" for initial assessments.
        
        EXAMPLE 1:
        Report: "40yo Male with 2 weeks of left side sciatica."
        Result: {{"demographics": {{"age": 40, "gender": "Male", "weight": "unknown"}}, "symptoms": ["left sided sciatica"], "history": "acute onset", "suggestedProcedure": "Physical Therapy"}}
        
        Clinical Report to Analyze:
        {report_text}
        
        {format_instructions}
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["report_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        full_prompt = prompt.format(report_text=report_text)
        response = await model.ainvoke(full_prompt)
        content = str(response.content)
        
        print(f"--- RAW NLP OUTPUT ---\n{content}\n---")
        
        try:
            raw_data = extract_json(content)
            
            def deep_clean(obj):
                if isinstance(obj, dict):
                    if "default" in obj: return deep_clean(obj["default"])
                    if "example" in obj: return deep_clean(obj["example"])
                    return {k: deep_clean(v) for k, v in obj.items() if k not in ["description", "type", "title"]}
                elif isinstance(obj, list):
                    return [deep_clean(i) for i in obj]
                return obj

            data = deep_clean(raw_data)
            if "properties" in data: data = data["properties"]

            # Field Mapping
            if "suggested_procedure" in data: data["suggestedProcedure"] = data.pop("suggested_procedure")
            if "medical_history" in data: data["history"] = data.pop("medical_history")

            demo = data.get("demographics", {})
            if not isinstance(demo, dict): demo = {}
            
            cleaned_data = {
                "demographics": Demographics(
                    age=demo.get("age"),
                    gender=demo.get("gender"),
                    weight=demo.get("weight")
                ),
                "symptoms": data.get("symptoms", []),
                "history": str(data.get("history", "Clinical history reviewed.")),
                "suggestedProcedure": str(data.get("suggestedProcedure", "Review clinically."))
            }

            if not isinstance(cleaned_data["symptoms"], list):
                cleaned_data["symptoms"] = [str(cleaned_data["symptoms"])]
            
            return ClinicalExtraction(**cleaned_data)
        except Exception as e:
            print(f"Recursive parsing for NLP failed: {e}")
            return ClinicalExtraction(
                demographics=Demographics(age=None, gender=None, weight=None),
                symptoms=[],
                history="Extraction parsed manually.",
                suggestedProcedure="Clinical review required."
            )
            
    except Exception as e:
        print(f"Critical Error in NLP Agent: {e}")
        return ClinicalExtraction(
            demographics=Demographics(age=None, gender=None, weight=None),
            symptoms=[], history="Error", suggestedProcedure="Review"
        )
