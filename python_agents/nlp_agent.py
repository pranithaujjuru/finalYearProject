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
    Extracts clinical data using a local Llama3.2 model via Ollama and LangChain.
    """
    try:
        # Initialize the local LLM
        model = ChatOllama(model="llama3.2")
        
        # Initialize the parser
        parser = PydanticOutputParser(pydantic_object=ClinicalExtraction)
        
        template = """
        You are a clinical NLP specialist. Extract patient demographics, symptoms, medical history, and suggest a baseline procedure from the provided clinical report.
        IMPORTANT: Return actual clinical data extracted from the text below. DO NOT return a JSON schema, field types, or placeholder descriptions.
        Return ONLY valid JSON.
        
        Clinical Report:
        {report_text}
        
        {format_instructions}
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["report_text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        # Invoke the model asynchronously
        full_prompt = prompt.format(report_text=report_text)
        response = await model.ainvoke(full_prompt)
        content = response.content
        
        # Debug logging
        print(f"--- RAW NLP OUTPUT ---\n{content}\n---")
        
        try:
            data = extract_json(content)
            
            # Unwrap if model returns schema-like structure
            if "properties" in data:
                data = data["properties"]
            
            # Map common snake_case to camelCase for local model resilience
            if "suggested_procedure" in data and "suggestedProcedure" not in data:
                data["suggestedProcedure"] = data.pop("suggested_procedure")
            
            # Robust field handling
            demographics_raw = data.get("demographics", {})
            if not isinstance(demographics_raw, dict): demographics_raw = {}
            
            cleaned_data = {
                "demographics": Demographics(
                    age=demographics_raw.get("age"),
                    gender=demographics_raw.get("gender"),
                    weight=demographics_raw.get("weight")
                ),
                "symptoms": data.get("symptoms", []),
                "history": data.get("history", "Detailed history not extracted."),
                "suggestedProcedure": data.get("suggestedProcedure", "Review clinically.")
            }
            
            # Protection against schema-dumping in symptoms/history
            if isinstance(cleaned_data["history"], str) and "'type': 'string'" in cleaned_data["history"]:
                cleaned_data["history"] = "Clinical history extracted from report."
            if isinstance(cleaned_data["suggestedProcedure"], str) and "'type': 'string'" in cleaned_data["suggestedProcedure"]:
                cleaned_data["suggestedProcedure"] = "Clinical review of findings."

            # Ensure symptoms is a list of strings
            if not isinstance(cleaned_data["symptoms"], list):
                cleaned_data["symptoms"] = [str(cleaned_data["symptoms"])]
            
            # Filter out schema-like strings in symptoms
            cleaned_data["symptoms"] = [str(s) for s in cleaned_data["symptoms"] if s is not None and "'type': 'string'" not in str(s)]
            
            return ClinicalExtraction(**cleaned_data)
        except Exception as e:
            print(f"Fallback parsing for NLP failed: {e}")
            return ClinicalExtraction(
                demographics=Demographics(age=None, gender=None, weight=None),
                symptoms=[],
                history=f"Automated extraction unsuccessful. Raw text: {content[:200]}...",
                suggestedProcedure="Clinical review required."
            )
            
    except Exception as e:
        print(f"Error in NLP Agent: {e}")
        return ClinicalExtraction(
            demographics=Demographics(age=None, gender=None, weight=None),
            symptoms=[],
            history="NLP extraction internal error",
            suggestedProcedure="Review clinically"
        )
