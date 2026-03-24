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

async def validate_prognosis(clinical_data: ClinicalExtraction, vision_finding: str) -> ValidationResult:
    """
    Validates the suggested procedure using a local RAG pipeline with Chroma and HuggingFace embeddings.
    """
    try:
        # Initialize the local LLM
        model = ChatOllama(model="llama3.2")
        
        # Initialize embeddings (local)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize the parser
        parser = PydanticOutputParser(pydantic_object=ValidationResult)
        
        # Load Chroma vector store (assumed to be at ./chroma_db)
        persist_directory = "./chroma_db"
        
        # Retrieval
        retrieved_docs = []
        if os.path.exists(persist_directory):
            try:
                vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
                search_query = f"{clinical_data.suggestedProcedure} {vision_finding} red flags contraindications"
                docs = vectorstore.similarity_search(search_query, k=3)
                retrieved_docs = [doc.page_content for doc in docs]
            except Exception:
                retrieved_docs = ["Error accessing ChromaDB."]
        else:
            retrieved_docs = ["Guideline DB not found. Using internal medical knowledge."]

        template = """
        Validate the suggested medical procedure against clinical guidelines and vision findings. 
        Flag any safety risks or 'red flags' (e.g., Cauda Equina Syndrome, progressive deficits).
        
        IMPORTANT: Your output must contain actual medical safety reasoning for THIS patient. 
        DO NOT return JSON schema definitions, regex patterns, or variable types.
        Return ONLY valid JSON.

        Clinical Data (JSON):
        {clinical_data}
        
        Vision Finding:
        {vision_finding}
        
        Retrieved Guideline Snippets:
        {guidelines}
        
        Your task is to provide a safety assessment, confidence score, identification of risks, and a final recommendation.
        
        {format_instructions}
        """
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["clinical_data", "vision_finding", "guidelines"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        context = "\n".join(retrieved_docs)
        full_prompt = prompt.format(
            clinical_data=clinical_data.json(),
            vision_finding=vision_finding,
            guidelines=context
        )
        
        # Invoke the model asynchronously
        response = await model.ainvoke(full_prompt)
        content = response.content
        
        # Debug logging
        print(f"--- RAW VALIDATOR OUTPUT ---\n{content}\n---")
        
        try:
            data = extract_json(content)
            
            # Unwrap if model returns schema-like structure
            if "properties" in data:
                data = data["properties"]
            
            # Map snake_case to camelCase for local model resilience
            mapping = {
                "is_safe": "isSafe",
                "confidence_score": "confidenceScore",
                "next_steps": "nextSteps",
                "referenced_guidelines": "referencedGuidelines"
            }
            for snake, camel in mapping.items():
                if snake in data and camel not in data:
                    data[camel] = data.pop(snake)
            
            # Use data.get with defaults for absolute resilience
            risks = data.get("risks", [])
            if not isinstance(risks, list): risks = [str(risks)]
            
            nextSteps = data.get("nextSteps", [])
            if not isinstance(nextSteps, list): nextSteps = [str(nextSteps)]
            
            referencedGuidelines = data.get("referencedGuidelines", [])
            if not isinstance(referencedGuidelines, list): referencedGuidelines = [str(referencedGuidelines)]
            
            cleaned_data = {
                "isSafe": bool(data.get("isSafe", True)),
                "confidenceScore": float(data.get("confidenceScore", 70.0)),
                "risks": [str(r) for r in risks if r is not None and "'type': 'string'" not in str(r)],
                "recommendation": str(data.get("recommendation", "Refer to clinical judgement.")),
                "nextSteps": [str(s) for s in nextSteps if s is not None and "'type': 'string'" not in str(s)],
                "referencedGuidelines": [str(g) for g in referencedGuidelines if g is not None and "'type': 'string'" not in str(g)]
            }
            
            # Sanity check for recommendation
            if "'type': 'string'" in cleaned_data["recommendation"]:
                cleaned_data["recommendation"] = "Clinical safety review required based on guideline data."
            
            return ValidationResult(**cleaned_data)
        except Exception as e:
            print(f"Fallback parsing for Validator failed: {e}")
            return ValidationResult(
                isSafe=True,
                confidenceScore=50.0,
                risks=["Automated safety check incomplete."],
                recommendation=f"Clinical review required. LLM Output: {content[:100]}...",
                nextSteps=["Manual guideline review"],
                referencedGuidelines=[]
            )

    except Exception as e:
        print(f"Error in Validator Agent: {e}")
        return ValidationResult(
            isSafe=True,
            confidenceScore=50.0,
            risks=[f"Validation agent internal error: {str(e)}"],
            recommendation="Please review clinical guidelines manually.",
            nextSteps=["Manual review required"],
            referencedGuidelines=[]
        )
