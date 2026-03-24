import { GoogleGenAI } from "@google/genai";
import { ClinicalExtraction, ClinicalExtractionSchema } from "../types/schemas.js";

export interface PatientQuestionnaire {
  painTrajectory: string;
  painDuration: string;
  previousTreatments: string;
  redFlags: string;
  neurologicalSymptoms: string;
}

export function synthesizePatientAnswers(answers: PatientQuestionnaire): string {
  return `Patient reports pain trajectory as ${answers.painTrajectory} lasting for ${answers.painDuration}. Previous treatments include: ${answers.previousTreatments}. Regarding red flags, patient states: ${answers.redFlags}. Patient describes neurological symptoms as: ${answers.neurologicalSymptoms}.`;
}

export async function processPatientIntake(answers: PatientQuestionnaire): Promise<ClinicalExtraction> {
  const narrativeText = synthesizePatientAnswers(answers);
  return extractClinicalData(narrativeText);
}

// Phase 2B: NLP Agent
// Extracts clinical data from raw text reports.

export async function extractClinicalData(reportText: string): Promise<ClinicalExtraction> {
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });
  
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: reportText,
    config: {
      systemInstruction: "You are a clinical NLP specialist. Extract patient demographics, symptoms, medical history, and suggest a baseline procedure from the provided clinical report. Output strictly in JSON.",
      responseMimeType: "application/json",
      responseSchema: {
        type: "object",
        properties: {
          demographics: {
            type: "object",
            properties: {
              age: { type: "number" },
              gender: { type: "string" },
              weight: { type: "string" }
            }
          },
          symptoms: { type: "array", items: { type: "string" } },
          history: { type: "string" },
          suggestedProcedure: { type: "string" },
        },
        required: ["demographics", "symptoms", "history", "suggestedProcedure"],
      },
    },
  });

  const result = JSON.parse(response.text || "{}");
  return ClinicalExtractionSchema.parse(result);
}
