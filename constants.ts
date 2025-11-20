
import { Tool, Language, Pattern, CodingTool } from './types';

export const TOOLS: Tool[] = [
  {
    id: 'langchain',
    name: 'LangChain',
    description: 'The most popular framework for developing applications powered by language models. Offers extensive integrations.',
    languages: [Language.Python, Language.NodeJS],
    tags: ['Framework', 'Orchestration', 'RAG'],
    supportedPatterns: ['explicit-function', 'structured-text', 'native-crdt', 'rag-agent'],
    githubStars: '80k+',
    complexity: 7,
    power: 9,
    website: 'https://langchain.com'
  },
  {
    id: 'autogen',
    name: 'AutoGen',
    description: 'A framework from Microsoft that enables the development of LLM applications using multiple agents that can converse with each other.',
    languages: [Language.Python, Language.NodeJS],
    tags: ['Multi-Agent', 'Microsoft', 'Conversation'],
    supportedPatterns: ['structured-text', 'explicit-function', 'shadow-workspace', 'multi-agent'],
    githubStars: '25k+',
    complexity: 8,
    power: 9,
    website: 'https://microsoft.github.io/autogen/'
  },
  {
    id: 'crewai',
    name: 'CrewAI',
    description: 'Cutting-edge framework for orchestrating role-playing, autonomous AI agents. Built on top of LangChain.',
    languages: [Language.Python],
    tags: ['Role-Playing', 'Task Delegation', 'High-Level'],
    supportedPatterns: ['structured-text', 'explicit-function', 'multi-agent'],
    githubStars: '15k+',
    complexity: 4,
    power: 7,
    website: 'https://crewai.com'
  },
  {
    id: 'llamaindex',
    name: 'LlamaIndex',
    description: 'A data framework for LLM applications to ingest, structure, and access private data.',
    languages: [Language.Python, Language.NodeJS],
    tags: ['Data', 'RAG', 'Indexing'],
    supportedPatterns: ['explicit-function', 'structured-text', 'rag-agent'],
    githubStars: '30k+',
    complexity: 6,
    power: 8,
    website: 'https://www.llamaindex.ai/'
  },
  {
    id: 'semantic-kernel',
    name: 'Semantic Kernel',
    description: 'SDK that integrates LLMs with conventional programming languages like C#, Python, and Java.',
    languages: [Language.Python],
    tags: ['Microsoft', 'Enterprise', 'Integration'],
    supportedPatterns: ['explicit-function'],
    githubStars: '18k+',
    complexity: 6,
    power: 8,
    website: 'https://github.com/microsoft/semantic-kernel'
  },
  {
    id: 'langgraph',
    name: 'LangGraph',
    description: 'A library for building stateful, multi-actor applications with LLMs, built on top of LangChain.',
    languages: [Language.Python, Language.NodeJS],
    tags: ['Stateful', 'Cyclic Graphs', 'Control Flow'],
    supportedPatterns: ['explicit-function', 'shadow-workspace', 'native-crdt', 'multi-agent'],
    githubStars: '5k+',
    complexity: 8,
    power: 10,
    website: 'https://langchain-ai.github.io/langgraph/'
  },
  {
    id: 'dspy',
    name: 'DSPy',
    description: 'A framework for algorithmically optimizing LM prompts and weights.',
    languages: [Language.Python],
    tags: ['Optimization', 'Research', 'Prompting'],
    supportedPatterns: ['structured-text'],
    githubStars: '10k+',
    complexity: 9,
    power: 9,
    website: 'https://dspy.ai'
  },
  {
    id: 'botpress',
    name: 'Botpress',
    description: 'A developer-first platform for building conversational AI. It has a visual builder but allows deep Node.js customization.',
    languages: [Language.NodeJS, Language.NoCode],
    tags: ['Platform', 'Visual Builder', 'Chatbot'],
    supportedPatterns: ['explicit-function'],
    githubStars: '12k+',
    complexity: 3,
    power: 7,
    website: 'https://botpress.com'
  },
  {
    id: 'aider',
    name: 'Aider',
    description: 'A command line tool that lets you pair program with LLMs, editing code in your local git repo.',
    languages: [Language.Python],
    tags: ['Coding', 'CLI', 'Git'],
    supportedPatterns: ['ast-semantic', 'shadow-workspace'],
    githubStars: '10k+',
    complexity: 5,
    power: 9,
    website: 'https://github.com/paul-gauthier/aider'
  }
];

export const PATTERNS: Pattern[] = [
  {
    id: 'multi-agent',
    name: 'Multi-Agent Orchestration',
    description: 'Multiple specialized agents (e.g., Coder, Reviewer, Planner) interact to solve complex tasks that a single model cannot handle alone.',
    useCase: 'Complex software development, Enterprise workflows, Simulation.',
    complexity: 'High',
    tags: ['Swarm', 'Hierarchical', 'Manager'],
    principles: 'Decomposition of thought. Instead of one massive prompt context, tasks are broken down. Agents communicate via message passing, often monitored by a "Manager" agent.',
    architecture: '1. **Router/Manager**: Analyzes the request and assigns it to a sub-agent.\n2. **Workers**: Specialized agents (with different system prompts/tools) execute tasks.\n3. **Handoffs**: Agents pass the state/result to the next agent in the chain.\n4. **Consolidation**: Manager compiles final output.',
    diagram: `graph TD
    User[User Request] --> Manager[Manager Agent]
    Manager -->|Delegate Coding| Coder[Coder Agent]
    Manager -->|Delegate Review| Reviewer[Reviewer Agent]
    Coder -->|PR Created| Reviewer
    Reviewer -->|Feedback| Coder
    Reviewer -->|Approved| Manager
    Manager -->|Final Response| User`,
    codeExample: `
# Using CrewAI (Python)
from crewai import Agent, Task, Crew

# Define specialized agents
researcher = Agent(
  role='Researcher',
  goal='Uncover groundbreaking technologies',
  backstory='You are a senior analyst...',
  tools=[search_tool]
)

writer = Agent(
  role='Writer',
  goal='Narrate compelling tech stories',
  backstory='You are a famous tech editor...',
)

# Define the tasks
task1 = Task(description='Research AI trends', agent=researcher)
task2 = Task(description='Write a blog post', agent=writer)

# Orchestrate
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  verbose=2
)

result = crew.kickoff()
`
  },
  {
    id: 'rag-agent',
    name: 'RAG Agent (Retrieval-Augmented)',
    description: 'An agent equipped with a "Retrieval" tool that allows it to search a vector database for relevant knowledge before generating an answer.',
    useCase: 'Customer support, Legal analysis, Documentation Q&A.',
    complexity: 'Medium',
    tags: ['Vector DB', 'Knowledge Base', 'Grounding'],
    principles: 'Grounding via context injection. The model is not trained on your private data; instead, relevant chunks of text are found via semantic similarity and pasted into the prompt at runtime.',
    architecture: '1. **Ingestion**: Documents are split into chunks and embedded into vectors.\n2. **Query**: User question is embedded.\n3. **Retrieval**: Top-k similar chunks are fetched from Vector DB.\n4. **Synthesis**: Chunks + Question are sent to LLM to generate answer.',
    diagram: `graph LR
    A[User Query] --> B(Embedding Model)
    B --> C{Vector DB}
    D[Documents] --> E[Chunking]
    E --> B
    C -->|Top-k Context| F[LLM Context Window]
    A --> F
    F --> G[Grounded Answer]`,
    codeExample: `
# Python (LangChain style conceptual)

# 1. Setup Retriever
vectorstore = Chroma.from_documents(documents, embedding_model)
retriever = vectorstore.as_retriever()

# 2. Define Agent Tool
@tool
def lookup_policy(query: str):
    """Useful for finding company policy details."""
    docs = retriever.get_relevant_documents(query)
    return "\\n".join([d.page_content for d in docs])

# 3. Initialize Agent
agent = create_openai_functions_agent(llm, [lookup_policy], prompt)
agent_executor = AgentExecutor(agent=agent, tools=[lookup_policy])

# 4. Run
agent_executor.invoke({"input": "What is the remote work policy?"})
`
  },
  {
    id: 'explicit-function',
    name: 'Explicit Function Calling',
    description: 'The LLM is trained to output specific tokens that map to executable code functions. The runtime intercepts these, executes the function, and feeds the result back.',
    useCase: 'API Integrations, Database Queries, Math calculations.',
    complexity: 'Low',
    tags: ['OpenAI', 'Tool Use', 'Deterministic'],
    principles: 'Relies on fine-tuned models (like GPT-4-Turbo or Claude 3.5 Sonnet) that understand a JSON schema definition of tools. The model pauses generation to request an action, waiting for the system to return the result.',
    architecture: '1. **Schema Definition**: Developer defines function signatures (name, args, docstring).\n2. **Inference**: LLM generates a "Tool Call" object instead of text.\n3. **Interception**: The Agent Runtime detects the stop sequence.\n4. **Execution**: Runtime executes the actual Python/JS function.\n5. **Recursion**: The result is appended to the chat history, and the LLM is invoked again to interpret the result.',
    diagram: `sequenceDiagram
    participant User
    participant AgentRuntime
    participant LLM
    participant Tool
    User->>AgentRuntime: "What is the weather in SF?"
    AgentRuntime->>LLM: Prompt + Tool Schemas
    LLM->>AgentRuntime: Call: get_weather(city="SF")
    AgentRuntime->>Tool: Execute get_weather
    Tool-->>AgentRuntime: {temp: 72, condition: "Sunny"}
    AgentRuntime->>LLM: Role: Tool, Content: {temp: 72...}
    LLM-->>User: "It is 72 degrees and Sunny."`,
    codeExample: `
# Python (OpenAI Native)
import json

def get_weather(location):
    return json.dumps({"location": location, "temp": "72F"})

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": "Weather in SF?"}],
    tools=tools
)
# The response.choices[0].message.tool_calls contains the call info
`
  },
  {
    id: 'structured-text',
    name: 'Structured Text Protocol (ReAct)',
    description: 'The agent follows a strict text-based format (like ReAct or XML) within the prompt to simulate reasoning and action steps without native API support.',
    useCase: 'Open source models, legacy systems, simple logic flows.',
    complexity: 'Medium',
    tags: ['ReAct', 'XML', 'Prompt Engineering'],
    principles: 'Uses few-shot prompting to enforce a "Thought -> Action -> Observation" loop. The model is instructed to output text in a specific format (e.g., `Thought: ... Action: ...`) which is then parsed by Regex.',
    architecture: '1. **System Prompt**: Contains strict instructions on output format (e.g., "Wrap actions in <action> tags").\n2. **Generation**: Model outputs a thought process followed by a structured command.\n3. **Parsing**: A Regex parser extracts the command from the text block.\n4. **Loop**: The output of the command is pasted back into the prompt context as an "Observation".',
    diagram: `graph TD
    A[User Input] --> B{Context Window}
    B --> C[LLM Generation]
    C --> D{Regex Parser}
    D -- Text Only --> E[Final Answer]
    D -- Match Found --> F[Extract Action]
    F --> G[Execute Code/API]
    G --> H[Observation String]
    H --> B`,
    codeExample: `
# Python Prompt Template Concept

prompt = """
Answer the following questions as best you can. 
You have access to the following tools: [Search, Calculator]

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Search, Calculator]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
"""
# The runtime then runs a while-loop, regex-matching "Action:"
`
  },
  {
    id: 'shadow-workspace',
    name: 'Shadow Workspace & Speculative Execution',
    description: 'The agent runs code in a sandboxed environment (Docker/WASM) to verify correctness before presenting it to the user or applying changes.',
    useCase: 'Code generation, Automated testing, Dangerous operations.',
    complexity: 'High',
    tags: ['Sandbox', 'Safety', 'Reliability'],
    principles: 'Iterative self-correction. The agent doesn\'t just guess code; it writes, runs, reads the stderr/stdout, fixes errors, and only returns the solution when it compiles/runs successfully.',
    architecture: '1. **Proposal**: LLM writes a code block.\n2. **Isolation**: Code is injected into a temporary container (Docker) or MicroVM (Firecracker).\n3. **Execution**: The environment runs the code/tests.\n4. **Feedback Loop**: If exit_code != 0, the error log is fed back to the LLM as a prompt to "Fix this error".\n5. **Commit**: Only successful code is applied to the main project.',
    diagram: `stateDiagram-v2
    [*] --> DraftCode
    DraftCode --> SandboxedRun
    SandboxedRun --> CheckResult
    CheckResult --> Success: Exit Code 0
    CheckResult --> AnalyzeError: Exit Code 1
    AnalyzeError --> RefineCode
    RefineCode --> SandboxedRun
    Success --> [*]`,
    codeExample: `
# Python (Conceptual using Docker SDK)
import docker

def run_speculative_code(code_snippet):
    client = docker.from_env()
    container = client.containers.run(
        "python:3.9-slim",
        command=f"python -c '{code_snippet}'",
        detach=True
    )
    result = container.wait()
    logs = container.logs()
    
    if result['StatusCode'] != 0:
        # Feed logs back to LLM for correction
        return prompt_llm_to_fix(code_snippet, logs)
    else:
        return code_snippet # Success!
`
  },
  {
    id: 'ast-semantic',
    name: 'AST-Based Semantic Perceptron',
    description: 'Instead of treating code as raw text, the agent interacts with the Abstract Syntax Tree (AST) to perform precise, syntax-aware edits.',
    useCase: 'Refactoring, Large Scale Codebases, Linter fixes.',
    complexity: 'High',
    tags: ['Compilers', 'Parsing', 'Tree-sitter'],
    principles: 'Prevents "drift" and syntax errors common in text-based editing. The agent identifies code not by line number (which changes) but by AST structure (Class -> Method -> Signature).',
    architecture: '1. **Parse**: Source code is parsed into an AST (using tools like Tree-sitter).\n2. **Locate**: LLM identifies the target node (e.g., "The function named update_user").\n3. **Transform**: Transformations are applied to the tree nodes directly.\n4. **Unparse**: The modified AST is converted back to source code, guaranteeing syntactic validity.',
    diagram: `graph LR
    A[Source Code] --> B[Parser]
    B --> C[Abstract Syntax Tree]
    C --> D[LLM Semantic Search]
    D --> E[Target Node Identified]
    E --> F[Tree Transformation]
    F --> G[Code Generator]
    G --> H[Valid Source Code]`,
    codeExample: `
# Python (using 'ast' module)
import ast

source = "def hello(): print('hi')"
tree = ast.parse(source)

# An Agent would identify this node via semantic search
class PrintModifier(ast.NodeTransformer):
    def visit_Call(self, node):
        # Change all print calls to 'logging.info'
        if isinstance(node.func, ast.Name) and node.func.id == 'print':
            node.func.id = 'logging.info'
        return node

new_tree = PrintModifier().visit(tree)
print(ast.unparse(new_tree))
# Output: def hello(): logging.info('hi')
`
  },
  {
    id: 'native-crdt',
    name: 'Native CRDT Integration',
    description: 'Conflict-free Replicated Data Types allow AI and Humans to edit the same state (documents, code, canvas) simultaneously without locking.',
    useCase: 'Collaborative editors (Google Docs style), Whiteboards, Shared State.',
    complexity: 'High',
    tags: ['Real-time', 'Collaboration', 'Distributed Systems'],
    principles: 'Treats AI as just another user in a distributed system. AI operations are commutative. Even if the AI acts on an old version of the document, its changes can be merged mathematically without conflict.',
    architecture: '1. **Data Structure**: State is stored in a CRDT (e.g., Y.js or Automerge).\n2. **Stream**: Changes are broadcast as small binary patches (deltas).\n3. **Integration**: The AI "Agent" connects via a WebSocket as a client.\n4. **Merges**: AI edits are applied locally and synced. If a human edits the same line simultaneously, the CRDT algorithm resolves the final state deterministically.',
    diagram: `sequenceDiagram
    participant Human
    participant Server
    participant AI_Agent
    Human->>Server: Insert "Hello" (Clock: 1)
    Server->>AI_Agent: Sync "Hello"
    par Simultaneous Edits
        Human->>Server: Insert " World" (Clock: 2)
        AI_Agent->>Server: Insert " Friend" (Clock: 2)
    end
    Server->>Human: Merge Result "Hello World Friend"
    Server->>AI_Agent: Merge Result "Hello World Friend"`,
    codeExample: `
// JavaScript (Y.js)
import * as Y from 'yjs'

const doc = new Y.Doc()
const ytext = doc.getText('codemirror')

// 1. Human Edits
ytext.insert(0, 'Hello ')

// 2. Agent Edits (simultaneous)
// The Agent calculates an insertion based on state
const agentOperation = () => {
   ytext.insert(6, 'World') 
}

// 3. CRDT handles the merge logic automatically
// independent of network latency or order.
`
  }
];

export const CODING_TOOLS: CodingTool[] = [
  {
    id: 'cursor',
    name: 'Cursor',
    type: 'IDE',
    coreMechanism: 'Shadow Workspace & RAG',
    description: 'A VS Code fork that integrates "Composer" (Agent) and "Tab" (FIM). It uses a shadow workspace to generate code diffs and RAG to index the entire codebase for context-aware answers.',
    features: ['Shadow Workspace (Composer)', 'Local/Remote Codebase Indexing', 'Fill-In-Middle Autocomplete'],
    website: 'https://cursor.com'
  },
  {
    id: 'windsurf',
    name: 'Windsurf (Codeium)',
    type: 'IDE',
    coreMechanism: 'Deep Context Flows (Cascades)',
    description: 'Built by Codeium. Features "Flows" which allow the agent to run terminal commands, read output, and edit files iteratively. It maintains a deep understanding of user intent via "Cascades".',
    features: ['Agentic Terminal Access', 'Deep Context Awareness', 'Iterative Refactoring'],
    website: 'https://codeium.com/windsurf'
  },
  {
    id: 'trae',
    name: 'Trae',
    type: 'IDE',
    coreMechanism: 'Data-Driven Context & Auto-Apply',
    description: 'ByteDance\'s AI IDE. Similar to Cursor/Windsurf, it focuses on high-speed code generation and intelligent context gathering from the project structure.',
    features: ['Intelligent Context', 'Chat-to-Code', 'Codebase Visualization'],
    website: 'https://www.trae.ai'
  },
  {
    id: 'zed',
    name: 'Zed',
    type: 'IDE',
    coreMechanism: 'CRDT & GPU Rendering',
    description: 'High-performance Rust-based editor. Uses CRDTs natively for collaboration, treating the AI as a collaborator in the document rather than a plugin wrapper.',
    features: ['Native CRDT', 'GPU Accelerated', 'Model Agnostic Chat'],
    website: 'https://zed.dev'
  },
  {
    id: 'aider',
    name: 'Aider',
    type: 'CLI',
    coreMechanism: 'AST-based "Repo Map"',
    description: 'A CLI tool that works with your local git repo. It generates a "Repo Map" (a compressed AST summary of your code) to fit the entire project structure into the LLM context window.',
    features: ['Git Integration', 'AST Repo Map', 'Terminal UI'],
    website: 'https://aider.chat'
  },
  {
    id: 'codebuddy',
    name: 'CodeBuddy',
    type: 'Extension',
    coreMechanism: 'Sandbox Execution',
    description: 'An agent that can not only write code but execute it in a sandbox to verify functionality before committing changes.',
    features: ['Sandboxed Execution', 'Self-Correction', 'Web Search'],
    website: 'https://codebuddy.ca'
  },
  {
    id: 'kiro',
    name: 'Kiro',
    type: 'IDE',
    coreMechanism: 'Agentic Workflow',
    description: 'A newer entrant focusing on autonomous agent workflows directly within the editor, automating repetitive refactoring tasks.',
    features: ['Task Automation', 'Agentic Refactoring'],
    website: 'https://kiro.ai'
  },
  {
    id: 'qoder',
    name: 'Qoder',
    type: 'Platform',
    coreMechanism: 'End-to-End Testing Agent',
    description: 'Focuses on ensuring code quality by generating and running tests (Shadow Workspace pattern) to validate generated code.',
    features: ['Test-Driven Generation', 'Quality Assurance'],
    website: 'https://qoder.ai'
  },
  {
    id: 'antigravity',
    name: 'Antigravity',
    type: 'IDE',
    coreMechanism: 'Visual Debugging & Object Tracking',
    description: 'Python-focused IDE that visualizes code execution flow. The AI uses this runtime data to understand bugs better than static analysis.',
    features: ['Time-Travel Debugging', 'Visual Flow', 'Runtime Analysis'],
    website: 'https://antigravity.ai' // Conceptual/Placeholder if niche
  },
  {
    id: 'codex-cli',
    name: 'OpenAI Codex / GitHub Copilot CLI',
    type: 'CLI',
    coreMechanism: 'Command Translation',
    description: 'Translates natural language into shell commands (Bash, PowerShell, Zsh). Uses specialized fine-tuning for shell syntax.',
    features: ['Natural Language to Bash', 'Command Explanation'],
    website: 'https://githubnext.com/projects/copilot-cli'
  },
  {
    id: 'cloud-code',
    name: 'Google Cloud Code',
    type: 'Extension',
    coreMechanism: 'Infrastructure aware (Gemini)',
    description: 'IDE extension for VS Code/IntelliJ. Uses Gemini to understand Kubernetes manifests, Terraform, and Cloud deployment configurations.',
    features: ['Kubernetes YAML Gen', 'Cloud Debugging', 'Duet AI Integration'],
    website: 'https://cloud.google.com/code'
  }
];

export const INITIAL_SYSTEM_INSTRUCTION = `
You are an expert Senior AI Engineer and Architect. 
Your goal is to assist developers in choosing the right stack for building AI Agents and understanding AI patterns.
You have deep knowledge of Python (LangChain, AutoGen, CrewAI, DSPy, etc.) and Node.js (LangChain.js, LlamaIndex.TS, etc.) ecosystems.
You are also an expert in advanced AI Agent architectures like AST-based editing, CRDTs for collaboration, and Speculative Execution.
You are well-versed in the modern AI IDE landscape, including Cursor (Shadow Workspace), Windsurf (Cascades), Aider (Repo Map), and Zed (CRDTs).
When asked, provide comparative analysis, code snippet examples in the requested language, and architectural advice.
Format your responses with clear Markdown.
`;
