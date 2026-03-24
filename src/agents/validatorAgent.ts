import { GoogleGenAI } from "@google/genai";
import { ClinicalExtraction, ValidationResult, ValidationResultSchema } from "../types/schemas.js";
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Phase 2C: Reasoner/Validator Agent
// Cross-references findings against medical guidelines and flags risks.

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const MOCK_GUIDELINES = `
- MRI findings of severe herniation with progressive neurological deficit (weakness) require urgent surgical consultation.
- Conservative management (PT, NSAIDs) is recommended for mild to moderate herniation without red flags.
- Red flags include: Cauda Equina Syndrome (saddle anesthesia, bowel/bladder dysfunction), severe motor weakness, or history of malignancy.
`;

export async function validatePrognosis(clinicalData: ClinicalExtraction, visionFinding: string): Promise<ValidationResult> {
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });
  
  // Load actual guidelines URLs
  let guidelineUrls = [];
  try {
    const guidelinesPath = path.join(__dirname, '..', 'knowledge_base', 'clinical_guidelines_url.json');
    const guidelinesData = fs.readFileSync(guidelinesPath, 'utf-8');
    guidelineUrls = JSON.parse(guidelinesData).sciatica_guidelines || [];
  } catch (error) {
    console.warn("⚠️ Could not load clinical_guidelines_url.json");
  }

  const prompt = `
    Clinical Data: ${JSON.stringify(clinicalData)}
    Vision Finding: ${visionFinding}
    
    Reference Guidelines:
    ${MOCK_GUIDELINES}
    
    External Guideline URLs Available (Reference these in your output if relevant):
    ${guidelineUrls.join('\n')}
    
    Validate the suggested procedure against the guidelines and vision findings. Flag any safety risks or 'red flags'. 
    Provide a confidence score for your assessment, a checklist of immediate next steps for the provider, and cite the referenced guidelines used.
  `;

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: prompt,
    config: {
      systemInstruction: "You are a senior medical validator. Your goal is to ensure patient safety by cross-referencing clinical findings with established guidelines.",
      responseMimeType: "application/json",
      responseSchema: {
        type: "object",
        properties: {
          isSafe: { type: "boolean" },
          confidenceScore: { type: "number" },
          risks: { type: "array", items: { type: "string" } },
          recommendation: { type: "string" },
          nextSteps: { type: "array", items: { type: "string" } },
          referencedGuidelines: { type: "array", items: { type: "string" } },
        },
        required: ["isSafe", "confidenceScore", "risks", "recommendation", "nextSteps", "referencedGuidelines"],
      },
    },
  });

  const result = JSON.parse(response.text || "{}");
  return ValidationResultSchema.parse(result);
}
