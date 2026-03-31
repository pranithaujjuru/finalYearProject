import base64
import json
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
import timm
from .schemas import VisionFinding
from .utils import extract_json

async def analyze_mri(image_bytes: bytes, mime_type: str) -> VisionFinding:
    """
    Analyzes a spinal MRI image using a local Llava model and a SOTA
    Swin Transformer (Large) classifier for medical precision.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        # --- 1. SOTA Medical Classification (Swin Transformer Large) ---
        specialist_finding = None
        # Updated SOTA path
        sota_model_path = os.path.join(base_dir, "../models/SOTA_Vision/swin_large_sciatica.pth")
        
        if os.path.exists(sota_model_path):
            try:
                # Setup Swin Architecture (Large, 384x384)
                model_sota = timm.create_model('swin_large_patch4_window12_384', pretrained=False)
                in_features = model_sota.head.fc.in_features
                model_sota.head.fc = nn.Sequential(
                    nn.Linear(in_features, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.3),
                    nn.Linear(512, 3) 
                )
                model_sota.load_state_dict(torch.load(sota_model_path, map_location=device))
                model_sota = model_sota.to(device)
                model_sota.eval()

                # SOTA Preprocess (384x384 for Swin-L)
                base_transform = transforms.Compose([
                    transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                # TTA Variations
                tta_transforms = [
                    lambda x: x, # Original
                    lambda x: transforms.functional.hflip(x), # Horizontal Flip
                    lambda x: transforms.functional.rotate(x, 10), # Slight Rotation +
                    lambda x: transforms.functional.rotate(x, -10), # Slight Rotation -
                    lambda x: transforms.functional.center_crop(transforms.functional.resize(x, 420), (384, 384)) # Zoom
                ]
                
                img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                tta_probs = []
                
                with torch.no_grad():
                    for tta_t in tta_transforms:
                        img_aug = tta_t(img_pil)
                        img_tensor = base_transform(img_aug).unsqueeze(0).to(device)
                        outputs = model_sota(img_tensor)
                        tta_probs.append(torch.softmax(outputs, dim=1))
                    
                    # SOTA Majority Voting / Prob Averaging
                    avg_probs = torch.mean(torch.stack(tta_probs), dim=0)
                    conf, pred = torch.max(avg_probs, 1)
                    
                    class_names = ["Herniated Disc", "No Stenosis", "Thecal Sac"]
                    if conf.item() > 0.45: # Swin-L is highly confident
                        specialist_finding = class_names[pred.item()]
            except Exception as e:
                print(f"SOTA specialist classifier error: {e}")
        else:
            print(f"Warning: SOTA model weights not found at {sota_model_path}")

        # Advanced Visual Localization (Llava:latest)
        llava_model = ChatOllama(model="llava:latest")
        
        # Base64 encode the image
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        
        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": """
                    [URGENT: OUTPUT ONLY DATA, NO SCHEMA]
                    You are a senior radiologist. Analyze this spinal MRI.
                    Identify:
                    1. FINDING: Main pathology summary.
                    2. SEVERITY: 'mild', 'moderate', or 'severe'.
                    3. LOCATION: Vertebral level (e.g., 'L4-L5').
                    
                    Return ONLY JSON: {"finding": "...", "severity": "...", "location": "..."}
                    """
                },
                {"type": "image_url", "image_url": f"data:{mime_type};base64,{base64_image}"}
            ]
        )
        
        response = await llava_model.ainvoke([message])
        content = str(response.content)
        
        try:
            data = extract_json(content)
            
            # Extract fields with robustness
            finding = data.get("finding", "Analysis complete.")
            severity = data.get("severity", "mild")
            location = data.get("location", "L-Spine")
            
            # --- 3. ENSEMBLE MERGE: Override LLM with Specialist Classifier ---
            if specialist_finding:
                finding = f"{specialist_finding} detected via specialized medical analysis. {finding}"
            
            return VisionFinding(finding=str(finding), severity=str(severity), location=str(location))
            
        except Exception as e:
            print(f"Fallback parsing for Vision: {e}")
            final_finding = specialist_finding if specialist_finding else "Visible features analyzed."
            return VisionFinding(finding=f"Detection: {final_finding}", severity="mild", location="L-Spine")
            
    except Exception as e:
        print(f"Critical Error in Vision Agent: {e}")
        return VisionFinding(finding="Vision analysis encountered an error.", severity="N/A", location="N/A")
