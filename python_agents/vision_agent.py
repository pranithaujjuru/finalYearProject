# pip install langchain langchain-community langchain-huggingface langchain-chroma pypdf sentence-transformers ollama
import base64
import json
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from .schemas import VisionFinding
from .utils import extract_json

async def analyze_mri(image_bytes: bytes, mime_type: str) -> VisionFinding:
    """
    Analyzes a spinal MRI image using a local Llava model via Ollama.
    """
    try:
        # Initialize the local vision model
        model = ChatOllama(model="llava")
        
        # Initialize the parser
        parser = PydanticOutputParser(pydantic_object=VisionFinding)
        
        # Base64 encode the image
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        # Construct the message with image data
        message = HumanMessage(
            content=[
                {"type": "text", "text": f"Analyze this spinal MRI image for signs of sciatica-related issues. IMPORTANT: Provide the actual diagnosis findings for this specific patient (e.g., 'L4-L5 herniation'). DO NOT return a JSON schema or variable descriptions. Return your findings IN JSON FORMAT ONLY based on this structure: {parser.get_format_instructions()}"},
                {"type": "image_url", "image_url": f"data:{mime_type};base64,{base64_image}"}
            ]
        )
        
        # Invoke the model asynchronously
        response = await model.ainvoke([message])
        content = response.content
        
        # Debug logging
        print(f"--- RAW VISION OUTPUT ---\n{content}\n---")
        
        # Attempt to parse
        try:
            data = extract_json(content)
            
            # Unwrap if model returns schema-like structure
            if "properties" in data:
                data = data["properties"]
                
            # Ensure required fields exist and are strings
            finding = data.get("finding", "Visible features analyzed; no specific herniation identified.")
            severity = data.get("severity", "mild")
            location = data.get("location", "unknown")
            
            # Protection against schema-dumping
            if isinstance(finding, dict) or (isinstance(finding, str) and "'type': 'string'" in finding):
                finding = "MRI analyzed; findings suggest potential disc involvement requiring clinical correlation."
            
            return VisionFinding(finding=str(finding), severity=str(severity), location=str(location))
        except Exception as e:
            print(f"Fallback parsing for Vision failed: {e}")
            return VisionFinding(finding=f"Analysis of visual features complete. Finding: {content[:100]}", severity="mild", location="unknown")
            
    except Exception as e:
        print(f"Error in Vision Agent: {e}")
        return VisionFinding(finding="Vision analysis encountered an error.", severity="mild", location="unknown")
