
import { GoogleGenAI, GenerateContentResponse } from "@google/genai";
import { INITIAL_SYSTEM_INSTRUCTION } from "../constants";
import { SupportedLang } from "../types";

const getClient = (): GoogleGenAI => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    throw new Error("API_KEY is not defined in the environment.");
  }
  return new GoogleGenAI({ apiKey });
};

export const sendMessageToGemini = async (
  message: string,
  history: { role: string; parts: { text: string }[] }[]
): Promise<string> => {
  try {
    const ai = getClient();
    
    // Transform simple history format to API format if needed, 
    // though creating a new chat with history is cleaner.
    const chat = ai.chats.create({
      model: 'gemini-2.5-flash',
      config: {
        systemInstruction: INITIAL_SYSTEM_INSTRUCTION,
        temperature: 0.7,
      },
      history: history.map(h => ({
        role: h.role,
        parts: h.parts
      }))
    });

    const result: GenerateContentResponse = await chat.sendMessage({
      message: message
    });

    return result.text || "I could not generate a response.";
  } catch (error) {
    console.error("Gemini API Error:", error);
    return "Sorry, I encountered an error while processing your request. Please check your API Key.";
  }
};

export const generateToolComparison = async (toolName: string, lang: SupportedLang = 'en'): Promise<string> => {
  try {
    const ai = getClient();
    const langMap = {
      'en': 'English',
      'zh': 'Chinese (Simplified)',
      'ja': 'Japanese'
    };
    const targetLang = langMap[lang];

    const prompt = `Provide a detailed technical breakdown of the AI tool "${toolName}". 
    Include: 
    1. Core Philosophy 
    2. Key Features 
    3. Best Use Case 
    4. A simple code snippet (Hello World equivalent) in its primary language.
    5. Pros and Cons.
    
    IMPORTANT: Output the entire response in ${targetLang}.
    Format as Markdown.`;

    const response = await ai.models.generateContent({
      model: 'gemini-2.5-flash',
      contents: prompt,
    });

    return response.text || "No details available.";
  } catch (error) {
    console.error("Gemini Details Error:", error);
    return "Failed to fetch details.";
  }
};
