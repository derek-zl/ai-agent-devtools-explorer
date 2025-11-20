
export type SupportedLang = 'en' | 'zh' | 'ja';

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
  description_zh?: string;
  description_ja?: string;
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
  name_zh?: string;
  name_ja?: string;
  description: string;
  description_zh?: string;
  description_ja?: string;
  useCase: string;
  useCase_zh?: string;
  useCase_ja?: string;
  complexity: 'Low' | 'Medium' | 'High' | 'Very High';
  tags: string[];
  // New fields for deep dive
  principles: string;
  principles_zh?: string;
  principles_ja?: string;
  architecture: string;
  architecture_zh?: string;
  architecture_ja?: string;
  diagram: string; // Mermaid graph definition
  codeExample: string; // New field for implementation example
}

export type CodingToolType = 'IDE' | 'CLI' | 'Extension' | 'Platform';

export interface CodingTool {
  id: string;
  name: string;
  type: CodingToolType;
  coreMechanism: string; // The architectural pattern used (e.g. Shadow Workspace)
  coreMechanism_zh?: string;
  coreMechanism_ja?: string;
  relatedPatternId?: string; // Link to the Pattern ID
  description: string;
  description_zh?: string;
  description_ja?: string;
  features: string[];
  features_zh?: string[];
  features_ja?: string[];
  website: string;
}

export interface BuilderExample {
  id: string;
  title: string;
  title_zh?: string;
  title_ja?: string;
  description: string;
  description_zh?: string;
  description_ja?: string;
  language: 'Python' | 'Node.js';
  difficulty: 'Beginner' | 'Intermediate';
  code: string;
  explanation: string;
  explanation_zh?: string;
  explanation_ja?: string;
}

export interface ChatMessage {
  role: 'user' | 'model';
  text: string;
  timestamp: number;
}

export type AnalysisState = 'idle' | 'loading' | 'success' | 'error';
