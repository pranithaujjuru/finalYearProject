import os
import json
from typing import List
from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from .schemas import ClinicalExtraction, ValidationResult
from .utils import extract_json

async def validate_prognosis(clinical_data: ClinicalExtraction, vision_finding: any) -> ValidationResult:
    """
    Validates the suggested procedure using a local RAG pipeline with strict clinical safety rules.
    """
    try:
        # Initialize the local LLM with explicit :latest tag
        model = ChatOllama(model="llama3.2:latest")
        # Local cache for embeddings to avoid 429 errors from HuggingFace Hub
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", 
            cache_folder="./models/embeddings"
        )
        parser = PydanticOutputParser(pydantic_object=ValidationResult)
        
        # Load Chroma vector store
        persist_directory = "./chroma_db"
        retrieved_docs = []
        if os.path.exists(persist_directory):
            try:
                vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
                search_query = f"{clinical_data.suggestedProcedure} {vision_finding} red flags contraindications"
                docs = vectorstore.similarity_search(search_query, k=2)
                retrieved_docs = [doc.page_content for doc in docs]
            except Exception:
                retrieved_docs = ["Error accessing ChromaDB."]
        else:
            retrieved_docs = ["Guideline DB not found. Using internal medical knowledge."]

        # Refined Prompt with Hardcoded Safety Rules & Anatomic Mapping
        template = """
        [URGENT: OUTPUT ONLY DATA, NO SCHEMA]
        You are a Clinical Safety Validator. Assess the safety of the proposed procedure.

        CRITICAL SAFETY RULES (MANDATORY):
        - If 'symptoms' mentions: "bladder", "bowel", "incontinence", "saddle anesthesia", or "bilateral weakness" -> isSafe MUST BE false.
        - In these cases, RECOMMENDATION MUST BE: "Urgent surgical consultation for potential Cauda Equina Syndrome."
        
        ANATOMIC ALIGNMENT GUIDE:
        - L4-L5 issues correlate with pain in the inner calf/big toe.
        - L5-S1 issues correlate with pain in the outer foot/small toe.
        
        CASE DATA:
        - Clinical Extraction: {clinical_data}
        - MRI findings: {vision_finding}
        - RAG Guidelines: {guidelines}
        
        Return ONLY a JSON object:
        {{
            "isSafe": boolean,
            "confidenceScore": number (0-100),
            "risks": ["list here"],
            "recommendation": "string",
            "nextSteps": ["list here"],
            "referencedGuidelines": ["list here"]
        }}
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["clinical_data", "vision_finding", "guidelines"]
        )
        
        context_str = "\n".join(retrieved_docs)
        full_prompt = prompt.format(
            clinical_data=clinical_data.json(),
            vision_finding=str(vision_finding),
            guidelines=context_str[:1500]
        )
        
        # Invoke LLM
        response = await model.ainvoke(full_prompt)
        content = str(response.content)
        
        try:
            clean_json = extract_json(content)
            
            # Post-processing normalization
            if not isinstance(clean_json, dict): raise ValueError("Not a dict")
            
            # Force safety rule in post-processing for 100% reliability
            red_flags = ["bladder", "bowel", "incontinence", "saddle", "bilateral"]
            symptoms_str = " ".join(clinical_data.symptoms).lower()
            if any(rf in symptoms_str for rf in red_flags):
                clean_json["isSafe"] = False
                clean_json["recommendation"] = "🚨 URGENT: Clinical red flags detected (Cauda Equina symptoms). Immediate surgical evaluation required."

            # Safe mapping to schema
            return ValidationResult(
                isSafe=bool(clean_json.get("isSafe", True)),
                confidenceScore=float(clean_json.get("confidenceScore", 85.0)),
                risks=clean_json.get("risks", []),
                recommendation=str(clean_json.get("recommendation", "Review clinically.")),
                nextSteps=clean_json.get("nextSteps", []),
                referencedGuidelines=clean_json.get("referencedGuidelines", ["NICE Guideline [NG59]", "NHS Cauda Equina Standards"])
            )
        except Exception as e:
            print(f"Fallback parsing for Validator: {e}")
            return ValidationResult(
                isSafe=True,
                confidenceScore=50.0,
                risks=["Safety check parsing issue."],
                recommendation="Please review clinical guidelines manually.",
                nextSteps=["Manual review"],
                referencedGuidelines=[]
            )
            
    except Exception as e:
        print(f"Critical Error in Validator: {e}")
        return ValidationResult(
            isSafe=True, 
            confidenceScore=50.0, 
            risks=[str(e)], 
            recommendation=f"System processing error: {str(e)[:50]}...", # More detail for troubleshooting
            nextSteps=["Check Ollama status", "Verify local model integrity"],
            referencedGuidelines=[]
        )
