
export enum Language {
  Python = 'Python',
  NodeJS = 'Node.js',
  Both = 'Both',
  NoCode = 'No-Code/Low-Code'
}

export interface Tool {
  id: string;
  name: string;
  description: string;
  languages: Language[];
  tags: string[];
  supportedPatterns: string[]; // Array of Pattern IDs
  githubStars?: string;
  complexity: number; // 1-10
  power: number; // 1-10
  website: string;
}

export interface Pattern {
  id: string;
  name: string;
  description: string;
  useCase: string;
  complexity: 'Low' | 'Medium' | 'High';
  tags: string[];
  // New fields for deep dive
  principles: string;
  architecture: string;
  diagram: string; // Mermaid graph definition
  codeExample: string; // New field for implementation example
}

export type CodingToolType = 'IDE' | 'CLI' | 'Extension' | 'Platform';

export interface CodingTool {
  id: string;
  name: string;
  type: CodingToolType;
  coreMechanism: string; // The architectural pattern used (e.g. Shadow Workspace)
  relatedPatternId?: string; // Link to the Pattern ID
  description: string;
  features: string[];
  website: string;
}

export interface BuilderExample {
  id: string;
  title: string;
  description: string;
  language: 'Python' | 'Node.js';
  difficulty: 'Beginner' | 'Intermediate';
  code: string;
  explanation: string;
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
  timestamp: number;
}

export type AnalysisState = 'idle' | 'loading' | 'success' | 'error';
