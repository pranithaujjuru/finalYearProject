import { GoogleGenAI } from "@google/genai";
import { VisionFinding, VisionFindingSchema } from "../types/schemas";

// Phase 2A: Vision Agent
// Uses Gemini 2.5 Flash for native vision analysis of MRI images.

export async function analyzeMRI(imageBuffer: Buffer, mimeType: string): Promise<VisionFinding> {
  const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });
  
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash-latest",
    contents: [
      {
        parts: [
          {
            inlineData: {
              data: imageBuffer.toString('base64'),
              mimeType: mimeType,
            },
          },
          {
            text: "Analyze this spinal MRI image for signs of sciatica-related issues like herniation, stenosis, or nerve compression. Output your findings in JSON format matching the schema: { finding: string, severity: 'mild'|'moderate'|'severe', location: string }",
          },
        ],
      },
    ],
    config: {
      responseMimeType: "application/json",
      responseSchema: {
        type: "object",
        properties: {
          finding: { type: "string" },
          severity: { type: "string", enum: ["mild", "moderate", "severe"] },
          location: { type: "string" },
        },
        required: ["finding", "severity", "location"],
      },
    },
  });

  const result = JSON.parse(response.text || "{}");
  return VisionFindingSchema.parse(result);
}
