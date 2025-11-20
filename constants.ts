
import { Tool, Language, Pattern, CodingTool, BuilderExample, SupportedLang } from './types';

export const UI_TEXT = {
  headerTitle: { en: "AI Agent Stack", zh: "AI Agent 技术栈", ja: "AI エージェント・スタック" },
  headerSubtitle: { en: "Explore the best tools, frameworks, and architectural patterns for building autonomous agents.", zh: "探索构建自主智能体的最佳工具、框架和架构模式。", ja: "自律型エージェントを構築するための最適なツール、フレームワーク、アーキテクチャパターンを探索します。" },
  navTools: { en: "Tools Explorer", zh: "工具浏览器", ja: "ツール一覧" },
  navPatterns: { en: "Agent Patterns", zh: "Agent 模式", ja: "エージェントパターン" },
  navIdes: { en: "AI IDEs", zh: "AI 编程工具", ja: "AI IDE" },
  navBuild: { en: "Build Guide", zh: "开发指南", ja: "構築ガイド" },
  searchPlaceholder: { en: "Search tools...", zh: "搜索工具...", ja: "ツールを検索..." },
  filterAll: { en: "All", zh: "全部", ja: "すべて" },
  viewGrid: { en: "Grid", zh: "网格", ja: "グリッド" },
  viewChart: { en: "Chart", zh: "图表", ja: "チャート" },
  complexity: { en: "Complexity", zh: "复杂度", ja: "複雑さ" },
  power: { en: "Power", zh: "能力", ja: "能力" },
  supportedPatterns: { en: "Supported Patterns", zh: "支持模式", ja: "対応パターン" },
  visitSite: { en: "Visit Official Site", zh: "访问官网", ja: "公式サイト" },
  primaryUseCase: { en: "Primary Use Case", zh: "主要应用场景", ja: "主なユースケース" },
  corePrinciples: { en: "Core Principles", zh: "核心原理", ja: "基本原則" },
  techArch: { en: "Technical Architecture", zh: "技术架构", ja: "技術アーキテクチャ" },
  implRef: { en: "Implementation Reference", zh: "实现参考", ja: "実装リファレンス" },
  comparisonGuide: { en: "Comparison & Selection Guide", zh: "对比与选型指南", ja: "比較と選択ガイド" },
  keyFeatures: { en: "Key Features", zh: "核心功能", ja: "主な機能" },
  whyWorks: { en: "Why this works", zh: "原理解析", ja: "仕組み" },
  logicBreakdown: { en: "Logic Breakdown", zh: "逻辑拆解", ja: "ロジック解説" },
  copyCode: { en: "Copy Code", zh: "复制代码", ja: "コードをコピー" },
  noToolsFound: { en: "No tools found matching your criteria.", zh: "没有找到匹配的工具。", ja: "条件に一致するツールが見つかりませんでした。" },
  clearFilters: { en: "Clear filters", zh: "清除筛选", ja: "フィルターをクリア" }
};

export const TOOLS: Tool[] = [
  {
    id: 'langchain',
    name: 'LangChain',
    description: 'The most popular framework for developing applications powered by language models. Offers extensive integrations.',
    description_zh: '开发大语言模型应用最流行的框架。提供广泛的集成组件。',
    description_ja: '大規模言語モデルを活用したアプリケーション開発で最も人気のあるフレームワーク。広範な統合機能を提供。',
    languages: [Language.Python, Language.NodeJS],
    tags: ['Framework', 'Orchestration', 'RAG'],
    supportedPatterns: ['explicit-function', 'structured-text', 'native-crdt', 'rag-agent', 'memory-augmented'],
    githubStars: '80k+',
    complexity: 7,
    power: 9,
    website: 'https://langchain.com'
  },
  {
    id: 'autogen',
    name: 'AutoGen',
    description: 'A framework from Microsoft that enables the development of LLM applications using multiple agents that can converse with each other.',
    description_zh: '微软推出的框架，支持开发多智能体相互对话协作的LLM应用。',
    description_ja: 'Microsoft発のフレームワーク。互いに会話できる複数のエージェントを使用したLLMアプリケーションの開発を可能にします。',
    languages: [Language.Python, Language.NodeJS],
    tags: ['Multi-Agent', 'Microsoft', 'Conversation'],
    supportedPatterns: ['structured-text', 'explicit-function', 'shadow-workspace', 'multi-agent', 'hierarchical', 'swarm'],
    githubStars: '25k+',
    complexity: 8,
    power: 9,
    website: 'https://microsoft.github.io/autogen/'
  },
  {
    id: 'crewai',
    name: 'CrewAI',
    description: 'Cutting-edge framework for orchestrating role-playing, autonomous AI agents. Built on top of LangChain.',
    description_zh: '用于编排角色扮演、自主AI智能体的前沿框架。基于LangChain构建。',
    description_ja: 'ロールプレイングを行う自律型AIエージェントを調整するための最先端フレームワーク。LangChain上に構築されています。',
    languages: [Language.Python],
    tags: ['Role-Playing', 'Task Delegation', 'High-Level'],
    supportedPatterns: ['structured-text', 'explicit-function', 'multi-agent', 'hierarchical'],
    githubStars: '15k+',
    complexity: 4,
    power: 7,
    website: 'https://crewai.com'
  },
  {
    id: 'llamaindex',
    name: 'LlamaIndex',
    description: 'A data framework for LLM applications to ingest, structure, and access private data.',
    description_zh: '用于LLM应用的数据框架，支持摄取、结构化和访问私有数据。',
    description_ja: 'LLMアプリケーションがプライベートデータを取り込み、構造化し、アクセスするためのデータフレームワーク。',
    languages: [Language.Python, Language.NodeJS],
    tags: ['Data', 'RAG', 'Indexing'],
    supportedPatterns: ['explicit-function', 'structured-text', 'rag-agent', 'memory-augmented'],
    githubStars: '30k+',
    complexity: 6,
    power: 8,
    website: 'https://www.llamaindex.ai/'
  },
  {
    id: 'semantic-kernel',
    name: 'Semantic Kernel',
    description: 'SDK that integrates LLMs with conventional programming languages like C#, Python, and Java.',
    description_zh: '将LLM与C#、Python和Java等传统编程语言集成的SDK。',
    description_ja: 'LLMをC#、Python、Javaなどの従来のプログラミング言語と統合するSDK。',
    languages: [Language.Python],
    tags: ['Microsoft', 'Enterprise', 'Integration'],
    supportedPatterns: ['explicit-function', 'plan-execute'],
    githubStars: '18k+',
    complexity: 6,
    power: 8,
    website: 'https://github.com/microsoft/semantic-kernel'
  },
  {
    id: 'langgraph',
    name: 'LangGraph',
    description: 'A library for building stateful, multi-actor applications with LLMs, built on top of LangChain.',
    description_zh: '基于LangChain构建的库，用于构建有状态、多参与者的LLM应用。',
    description_ja: 'LangChain上に構築された、ステートフルでマルチアクターなLLMアプリケーションを構築するためのライブラリ。',
    languages: [Language.Python, Language.NodeJS],
    tags: ['Stateful', 'Cyclic Graphs', 'Control Flow'],
    supportedPatterns: ['explicit-function', 'shadow-workspace', 'native-crdt', 'multi-agent', 'state-machine', 'human-in-the-loop'],
    githubStars: '5k+',
    complexity: 8,
    power: 10,
    website: 'https://langchain-ai.github.io/langgraph/'
  },
  {
    id: 'dspy',
    name: 'DSPy',
    description: 'A framework for algorithmically optimizing LM prompts and weights.',
    description_zh: '一个通过算法优化语言模型提示词和权重的框架。',
    description_ja: 'LMのプロンプトと重みをアルゴリズム的に最適化するためのフレームワーク。',
    languages: [Language.Python],
    tags: ['Optimization', 'Research', 'Prompting'],
    supportedPatterns: ['structured-text', 'cot', 'self-reflection'],
    githubStars: '10k+',
    complexity: 9,
    power: 9,
    website: 'https://dspy.ai'
  },
  {
    id: 'botpress',
    name: 'Botpress',
    description: 'A developer-first platform for building conversational AI. It has a visual builder but allows deep Node.js customization.',
    description_zh: '开发者优先的对话式AI构建平台。拥有可视化构建器，同时也支持深入的Node.js定制。',
    description_ja: '対話型AIを構築するための開発者優先プラットフォーム。ビジュアルビルダーを備えていますが、Node.jsによる詳細なカスタマイズも可能です。',
    languages: [Language.NodeJS, Language.NoCode],
    tags: ['Platform', 'Visual Builder', 'Chatbot'],
    supportedPatterns: ['explicit-function', 'state-machine'],
    githubStars: '12k+',
    complexity: 3,
    power: 7,
    website: 'https://botpress.com'
  },
  {
    id: 'aider',
    name: 'Aider',
    description: 'A command line tool that lets you pair program with LLMs, editing code in your local git repo.',
    description_zh: '一个命令行工具，让你能与LLM结对编程，直接编辑本地Git仓库中的代码。',
    description_ja: 'LLMとペアプログラミングを行い、ローカルのGitリポジトリ内のコードを編集できるコマンドラインツール。',
    languages: [Language.Python],
    tags: ['Coding', 'CLI', 'Git'],
    supportedPatterns: ['ast-semantic', 'shadow-workspace', 'self-reflection'],
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
    name_zh: '多智能体编排',
    name_ja: 'マルチエージェント・オーケストレーション',
    description: 'Multiple specialized agents (e.g., Coder, Reviewer, Planner) interact to solve complex tasks that a single model cannot handle alone.',
    description_zh: '多个专用智能体（如编码员、审核员、规划员）相互交互，解决单个模型无法独立处理的复杂任务。',
    description_ja: '複数の専門化されたエージェント（コーダー、レビュアー、プランナーなど）が相互作用し、単一モデルでは処理できない複雑なタスクを解決します。',
    useCase: 'Complex software development, Enterprise workflows, Simulation.',
    useCase_zh: '复杂软件开发，企业工作流，仿真模拟。',
    useCase_ja: '複雑なソフトウェア開発、エンタープライズワークフロー、シミュレーション。',
    complexity: 'High',
    tags: ['Swarm', 'Hierarchical', 'Manager'],
    principles: 'Decomposition of thought. Instead of one massive prompt context, tasks are broken down. Agents communicate via message passing, often monitored by a "Manager" agent.',
    principles_zh: '思维分解。任务被拆解，而不是仅仅依赖一个巨大的提示词上下文。智能体通过消息传递进行通信，通常由"经理"智能体监控。',
    principles_ja: '思考の分解。巨大なプロンプトコンテキストではなく、タスクが分解されます。エージェントはメッセージパッシングを通じて通信し、多くの場合「マネージャー」エージェントによって監視されます。',
    architecture: '1. **Router/Manager**: Analyzes the request and assigns it to a sub-agent.\n2. **Workers**: Specialized agents (with different system prompts/tools) execute tasks.\n3. **Handoffs**: Agents pass the state/result to the next agent in the chain.\n4. **Consolidation**: Manager compiles final output.',
    architecture_zh: '1. **路由/经理**：分析请求并分配给子智能体。\n2. **工人**：专用智能体（具有不同的系统提示/工具）执行任务。\n3. **交接**：智能体将状态/结果传递给链中的下一个智能体。\n4. **整合**：经理汇总最终输出。',
    architecture_ja: '1. **ルーター/マネージャー**: リクエストを分析し、サブエージェントに割り当てます。\n2. **ワーカー**: 専門化されたエージェント（異なるシステムプロンプト/ツールを持つ）がタスクを実行します。\n3. **ハンドオフ**: エージェントは状態/結果をチェーン内の次のエージェントに渡します。\n4. **統合**: マネージャーが最終出力をまとめます。',
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
    name_zh: 'RAG 智能体 (检索增强)',
    name_ja: 'RAG エージェント (検索拡張)',
    description: 'An agent equipped with a "Retrieval" tool that allows it to search a vector database for relevant knowledge before generating an answer.',
    description_zh: '配备"检索"工具的智能体，能够在生成答案之前在向量数据库中搜索相关知识。',
    description_ja: '「検索」ツールを備えたエージェントで、回答を生成する前に関連知識をベクトルデータベースから検索できます。',
    useCase: 'Customer support, Legal analysis, Documentation Q&A.',
    useCase_zh: '客户支持，法律分析，文档问答。',
    useCase_ja: 'カスタマーサポート、法的分析、ドキュメントQ&A。',
    complexity: 'Medium',
    tags: ['Vector DB', 'Knowledge Base', 'Grounding'],
    principles: 'Grounding via context injection. The model is not trained on your private data; instead, relevant chunks of text are found via semantic similarity and pasted into the prompt at runtime.',
    principles_zh: '通过上下文注入实现基于事实的回答。模型未在私有数据上进行训练；相反，通过语义相似性找到相关文本块，并在运行时粘贴到提示词中。',
    principles_ja: 'コンテキスト注入によるグラウンディング。モデルはプライベートデータでトレーニングされていません。代わりに、意味的類似性を介して関連するテキストチャンクが見つかり、実行時にプロンプトに貼り付けられます。',
    architecture: '1. **Ingestion**: Documents are split into chunks and embedded into vectors.\n2. **Query**: User question is embedded.\n3. **Retrieval**: Top-k similar chunks are fetched from Vector DB.\n4. **Synthesis**: Chunks + Question are sent to LLM to generate answer.',
    architecture_zh: '1. **摄入**：文档被分割成块并嵌入为向量。\n2. **查询**：用户问题被嵌入。\n3. **检索**：从向量数据库中获取Top-k相似块。\n4. **合成**：块+问题被发送给LLM以生成答案。',
    architecture_ja: '1. **取り込み**: ドキュメントがチャンクに分割され、ベクトルに埋め込まれます。\n2. **クエリ**: ユーザーの質問が埋め込まれます。\n3. **検索**: 類似性の高い上位k個のチャンクがベクトルDBから取得されます。\n4. **合成**: チャンクと質問がLLMに送信され、回答が生成されます。',
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
    name_zh: '显式函数调用',
    name_ja: '明示的関数呼び出し',
    description: 'The LLM is trained to output specific tokens that map to executable code functions. The runtime intercepts these, executes the function, and feeds the result back.',
    description_zh: 'LLM被训练输出映射到可执行代码函数的特定token。运行时拦截这些token，执行函数，并将结果反馈回去。',
    description_ja: 'LLMは、実行可能なコード関数にマップされる特定のトークンを出力するようにトレーニングされています。ランタイムはこれらをインターセプトして関数を実行し、結果をフィードバックします。',
    useCase: 'API Integrations, Database Queries, Math calculations.',
    useCase_zh: 'API集成，数据库查询，数学计算。',
    useCase_ja: 'API統合、データベースクエリ、数学計算。',
    complexity: 'Low',
    tags: ['OpenAI', 'Tool Use', 'Deterministic'],
    principles: 'Relies on fine-tuned models (like GPT-4-Turbo or Claude 3.5 Sonnet) that understand a JSON schema definition of tools. The model pauses generation to request an action, waiting for the system to return the result.',
    principles_zh: '依赖于微调模型（如GPT-4-Turbo或Claude 3.5 Sonnet），这些模型理解工具的JSON架构定义。模型暂停生成以请求操作，等待系统返回结果。',
    principles_ja: 'ツールのJSONスキーマ定義を理解するファインチューニングされたモデル（GPT-4-TurboやClaude 3.5 Sonnetなど）に依存します。モデルは生成を一時停止してアクションを要求し、システムが結果を返すのを待ちます。',
    architecture: '1. **Schema Definition**: Developer defines function signatures (name, args, docstring).\n2. **Inference**: LLM generates a "Tool Call" object instead of text.\n3. **Interception**: The Agent Runtime detects the stop sequence.\n4. **Execution**: Runtime executes the actual Python/JS function.\n5. **Recursion**: The result is appended to the chat history, and the LLM is invoked again to interpret the result.',
    architecture_zh: '1. **架构定义**：开发者定义函数签名（名称、参数、文档字符串）。\n2. **推理**：LLM生成"工具调用"对象而非文本。\n3. **拦截**：智能体运行时检测停止序列。\n4. **执行**：运行时执行实际的Python/JS函数。\n5. **递归**：结果被追加到聊天历史中，LLM再次被调用以解释结果。',
    architecture_ja: '1. **スキーマ定義**: 開発者が関数シグネチャ（名前、引数、docstring）を定義します。\n2. **推論**: LLMはテキストの代わりに「ツール呼び出し」オブジェクトを生成します。\n3. **インターセプト**: エージェントランタイムが停止シーケンスを検出します。\n4. **実行**: ランタイムが実際のPython/JS関数を実行します。\n5. **再帰**: 結果がチャット履歴に追加され、LLMが再度呼び出されて結果を解釈します。',
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
    name_zh: '结构化文本协议 (ReAct)',
    name_ja: '構造化テキストプロトコル (ReAct)',
    description: 'The agent follows a strict text-based format (like ReAct or XML) within the prompt to simulate reasoning and action steps without native API support.',
    description_zh: '智能体在提示词中遵循严格的基于文本的格式（如ReAct或XML），以在没有原生API支持的情况下模拟推理和行动步骤。',
    description_ja: 'エージェントはプロンプト内で厳密なテキストベースの形式（ReActやXMLなど）に従い、ネイティブAPIサポートなしで推論とアクションの手順をシミュレートします。',
    useCase: 'Open source models, legacy systems, simple logic flows.',
    useCase_zh: '开源模型，遗留系统，简单逻辑流。',
    useCase_ja: 'オープンソースモデル、レガシーシステム、単純なロジックフロー。',
    complexity: 'Medium',
    tags: ['ReAct', 'XML', 'Prompt Engineering'],
    principles: 'Uses few-shot prompting to enforce a "Thought -> Action -> Observation" loop. The model is instructed to output text in a specific format (e.g., `Thought: ... Action: ...`) which is then parsed by Regex.',
    principles_zh: '使用少样本提示来强制执行"思考 -> 行动 -> 观察"循环。模型被指示以特定格式（例如 `Thought: ... Action: ...`）输出文本，然后由正则解析。',
    principles_ja: '「思考 -> アクション -> 観察」ループを強制するために、few-shotプロンプティングを使用します。モデルは特定の形式（例：`Thought: ... Action: ...`）でテキストを出力するように指示され、正規表現で解析されます。',
    architecture: '1. **System Prompt**: Contains strict instructions on output format (e.g., "Wrap actions in <action> tags").\n2. **Generation**: Model outputs a thought process followed by a structured command.\n3. **Parsing**: A Regex parser extracts the command from the text block.\n4. **Loop**: The output of the command is pasted back into the prompt context as an "Observation".',
    architecture_zh: '1. **系统提示**：包含关于输出格式的严格指令。\n2. **生成**：模型输出思考过程，后跟结构化命令。\n3. **解析**：正则解析器从文本块中提取命令。\n4. **循环**：命令的输出作为"观察"粘贴回提示上下文。',
    architecture_ja: '1. **システムプロンプト**: 出力形式に関する厳密な指示が含まれています。\n2. **生成**: モデルは思考プロセスとそれに続く構造化コマンドを出力します。\n3. **解析**: 正規表現パーサーがテキストブロックからコマンドを抽出します。\n4. **ループ**: コマンドの出力は「観察」としてプロンプトコンテキストに貼り付けられます。',
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
def react_agent(query):
    while not task_complete:
        # 1. Thought: Reason about the next step
        thought = llm.generate(f"Question: {query}\\nThought:")
        
        # 2. Action: Execute concrete operation
        action = llm.generate(f"Thought: {thought}\\nAction:")
        result = execute_action(action)
        
        # 3. Observation: Observe result and continue
        observation = result
        query = f"Thought: {thought}\\nAction: {action}\\nObservation: {observation}"
`
  },
  {
    id: 'shadow-workspace',
    name: 'Shadow Workspace & Speculative Execution',
    name_zh: '影子工作区与推测执行',
    name_ja: 'シャドウワークスペースと投機的実行',
    description: 'The agent runs code in a sandboxed environment (Docker/WASM) to verify correctness before presenting it to the user or applying changes.',
    description_zh: '智能体在沙盒环境（Docker/WASM）中运行代码，在向用户展示或应用更改之前验证正确性。',
    description_ja: 'エージェントはサンドボックス環境（Docker/WASM）でコードを実行し、ユーザーに提示したり変更を適用したりする前に正しさを検証します。',
    useCase: 'Code generation, Automated testing, Dangerous operations.',
    useCase_zh: '代码生成，自动化测试，危险操作。',
    useCase_ja: 'コード生成、自動テスト、危険な操作。',
    complexity: 'High',
    tags: ['Sandbox', 'Safety', 'Reliability'],
    principles: 'Iterative self-correction. The agent doesn\'t just guess code; it writes, runs, reads the stderr/stdout, fixes errors, and only returns the solution when it compiles/runs successfully.',
    principles_zh: '迭代自我修正。智能体不仅仅是猜测代码；它编写、运行、读取标准错误/输出、修复错误，只有在编译/运行成功时才返回解决方案。',
    principles_ja: '反復的な自己修正。エージェントは単にコードを推測するのではなく、書き、実行し、stderr/stdoutを読み、エラーを修正し、コンパイル/実行が成功した場合にのみソリューションを返します。',
    architecture: '1. **Proposal**: LLM writes a code block.\n2. **Isolation**: Code is injected into a temporary container (Docker) or MicroVM (Firecracker).\n3. **Execution**: The environment runs the code/tests.\n4. **Feedback Loop**: If exit_code != 0, the error log is fed back to the LLM as a prompt to "Fix this error".\n5. **Commit**: Only successful code is applied to the main project.',
    architecture_zh: '1. **提议**：LLM编写代码块。\n2. **隔离**：代码注入临时容器（Docker）或MicroVM。\n3. **执行**：环境运行代码/测试。\n4. **反馈循环**：如果退出代码!=0，错误日志反馈给LLM以"修复此错误"。\n5. **提交**：只有成功的代码才应用到主项目。',
    architecture_ja: '1. **提案**: LLMがコードブロックを記述します。\n2. **分離**: コードは一時コンテナ（Docker）またはMicroVMに注入されます。\n3. **実行**: 環境がコード/テストを実行します。\n4. **フィードバックループ**: 終了コード!= 0の場合、エラーログが「このエラーを修正せよ」というプロンプトとしてLLMにフィードバックされます。\n5. **コミット**: 成功したコードのみがメインプロジェクトに適用されます。',
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
    name_zh: '基于AST的语义感知机',
    name_ja: 'ASTベースの意味論的パーセプトロン',
    description: 'Instead of treating code as raw text, the agent interacts with the Abstract Syntax Tree (AST) to perform precise, syntax-aware edits.',
    description_zh: '智能体不将代码视为原始文本，而是与抽象语法树（AST）交互，以执行精确的、语法感知的编辑。',
    description_ja: 'コードを生のテキストとして扱うのではなく、エージェントは抽象構文木（AST）と対話して、正確で構文を意識した編集を実行します。',
    useCase: 'Refactoring, Large Scale Codebases, Linter fixes.',
    useCase_zh: '重构，大规模代码库，Linter修复。',
    useCase_ja: 'リファクタリング、大規模コードベース、リンター修正。',
    complexity: 'High',
    tags: ['Compilers', 'Parsing', 'Tree-sitter'],
    principles: 'Prevents "drift" and syntax errors common in text-based editing. The agent identifies code not by line number (which changes) but by AST structure (Class -> Method -> Signature).',
    principles_zh: '防止基于文本的编辑中常见的"漂移"和语法错误。智能体不通过行号（会变化）而是通过AST结构（类 -> 方法 -> 签名）识别代码。',
    principles_ja: 'テキストベースの編集で一般的な「ドリフト」や構文エラーを防ぎます。エージェントは行番号（変更される）ではなく、AST構造（クラス -> メソッド -> シグネチャ）によってコードを識別します。',
    architecture: '1. **Parse**: Source code is parsed into an AST (using tools like Tree-sitter).\n2. **Locate**: LLM identifies the target node (e.g., "The function named update_user").\n3. **Transform**: Transformations are applied to the tree nodes directly.\n4. **Unparse**: The modified AST is converted back to source code, guaranteeing syntactic validity.',
    architecture_zh: '1. **解析**：源代码被解析为AST。\n2. **定位**：LLM识别目标节点。\n3. **转换**：直接对树节点应用转换。\n4. **反解析**：修改后的AST转换回源代码，保证语法有效性。',
    architecture_ja: '1. **解析**: ソースコードがASTに解析されます。\n2. **検索**: LLMがターゲットノードを特定します。\n3. **変換**: 変換がツリーノードに直接適用されます。\n4. **逆解析**: 変更されたASTがソースコードに変換され、構文の妥当性が保証されます。',
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
    name_zh: '原生 CRDT 协作流',
    name_ja: 'ネイティブCRDT統合',
    description: 'Conflict-free Replicated Data Types allow AI and Humans to edit the same state (documents, code, canvas) simultaneously without locking.',
    description_zh: '无冲突复制数据类型允许AI和人类同时编辑同一状态（文档、代码、画布）而无需锁定。',
    description_ja: '競合のない複製データ型により、AIと人間がロックなしで同じ状態（ドキュメント、コード、キャンバス）を同時に編集できます。',
    useCase: 'Collaborative editors (Google Docs style), Whiteboards, Shared State.',
    useCase_zh: '协作编辑器（Google Docs风格），白板，共享状态。',
    useCase_ja: '共同編集エディタ（Google Docsスタイル）、ホワイトボード、共有状態。',
    complexity: 'High',
    tags: ['Real-time', 'Collaboration', 'Distributed Systems'],
    principles: 'Treats AI as just another user in a distributed system. AI operations are commutative. Even if the AI acts on an old version of the document, its changes can be merged mathematically without conflict.',
    principles_zh: '将AI视为分布式系统中的另一个用户。AI操作是可交换的。即使AI对旧版本的文档进行操作，其更改也可以在数学上无冲突地合并。',
    principles_ja: '分散システムの単なる別のユーザーとしてAIを扱います。AI操作は可換です。AIがドキュメントの古いバージョンで動作しても、その変更は競合することなく数学的にマージできます。',
    architecture: '1. **Data Structure**: State is stored in a CRDT (e.g., Y.js or Automerge).\n2. **Stream**: Changes are broadcast as small binary patches (deltas).\n3. **Integration**: The AI "Agent" connects via a WebSocket as a client.\n4. **Merges**: AI edits are applied locally and synced. If a human edits the same line simultaneously, the CRDT algorithm resolves the final state deterministically.',
    architecture_zh: '1. **数据结构**：状态存储在CRDT中。\n2. **流**：更改作为小的二进制补丁广播。\n3. **集成**：AI"智能体"作为客户端通过WebSocket连接。\n4. **合并**：AI编辑在本地应用并同步。如果人类同时编辑同一行，CRDT算法确定性地解决最终状态。',
    architecture_ja: '1. **データ構造**: 状態はCRDTに保存されます。\n2. **ストリーム**: 変更は小さなバイナリパッチとしてブロードキャストされます。\n3. **統合**: AI「エージェント」はWebSocket経由でクライアントとして接続します。\n4. **マージ**: AIの編集はローカルに適用され、同期されます。人間が同時に同じ行を編集した場合、CRDTアルゴリズムが最終状態を決定的に解決します。',
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
  },
  {
    id: 'plan-execute',
    name: 'Plan-and-Execute',
    name_zh: '计划与执行模式',
    name_ja: '計画と実行パターン',
    description: 'The agent first formulates a detailed, multi-step plan before starting execution. It then executes steps sequentially, often with a check/re-plan phase.',
    description_zh: '智能体在开始执行前首先制定详细的多步骤计划。然后按顺序执行步骤，通常带有检查/重新计划阶段。',
    description_ja: 'エージェントは実行を開始する前に、まず詳細なマルチステップ計画を作成します。その後、多くの場合チェック/再計画フェーズを伴いながら、ステップを順次実行します。',
    useCase: 'Long-term research, Project management, Complex workflows.',
    useCase_zh: '长期研究，项目管理，复杂工作流。',
    useCase_ja: '長期的な調査、プロジェクト管理、複雑なワークフロー。',
    complexity: 'High',
    tags: ['Planning', 'Workflow', 'Management'],
    principles: 'Separates "Thinking" (Planning) from "Doing" (Execution). This reduces the risk of the agent getting lost in the weeds of a single step and forgetting the overall goal.',
    principles_zh: '将"思考"（计划）与"行动"（执行）分离。这降低了智能体迷失在单个步骤细节中而忘记整体目标的风险。',
    principles_ja: '「思考」（計画）と「実行」（実行）を分離します。これにより、エージェントが単一のステップの細部で迷子になり、全体的な目標を忘れるリスクが軽減されます。',
    architecture: '1. **Planner**: LLM generates a numbered list of steps.\n2. **Executor**: A loop picks the next step and executes it using tools.\n3. **Replanner**: After each step, the agent checks if the original plan is still valid or needs adjustment based on new data.',
    architecture_zh: '1. **规划器**：LLM生成编号步骤列表。\n2. **执行器**：循环选取下一步并使用工具执行。\n3. **重规划器**：每一步后，智能体检查原计划是否仍然有效或需调整。',
    architecture_ja: '1. **プランナー**: LLMが番号付きのステップリストを生成します。\n2. **実行者**: ループが次のステップを選択し、ツールを使用して実行します。\n3. **再プランナー**: 各ステップの後、エージェントは元の計画がまだ有効か、新しいデータに基づいて調整が必要かを確認します。',
    diagram: `graph TD
    Goal[User Goal] --> Planner
    Planner -->|Generate Plan| PlanList[Step 1, Step 2, Step 3...]
    PlanList --> Executor
    Executor -->|Execute Step 1| Result1
    Result1 -->|Success| Executor
    Result1 -->|Fail| Replan[Replanner]
    Replan --> PlanList`,
    codeExample: `
class PlanExecuteAgent:
    def plan(self, goal):
        # Generate detailed plan
        plan = llm.generate(f"Goal: {goal}\\nGenerate step-by-step plan:")
        return parse_plan(plan)
    
    def execute(self, plan):
        results = []
        for step in plan:
            result = self.execute_step(step)
            results.append(result)
            # Dynamic adjustment
            if need_replan:
                plan = self.adjust_plan(plan, results)
        return results
`
  },
  {
    id: 'cot',
    name: 'Chain-of-Thought (CoT)',
    name_zh: '思维链 (CoT)',
    name_ja: '思考の連鎖 (CoT)',
    description: 'Forces the model to verbalize its reasoning process ("think step-by-step") before giving a final answer.',
    description_zh: '强制模型在给出最终答案之前口头表达其推理过程（"一步一步地思考"）。',
    description_ja: '最終的な答えを出す前に、モデルに推論プロセスを言語化（「ステップバイステップで考える」）させます。',
    useCase: 'Math problems, Logical puzzles, Complex reasoning.',
    useCase_zh: '数学问题，逻辑谜题，复杂推理。',
    useCase_ja: '数学の問題、論理パズル、複雑な推論。',
    complexity: 'Low',
    tags: ['Reasoning', 'Logic', 'Prompting'],
    principles: 'Large Language Models perform significantly better on logic tasks when they are allowed to generate intermediate reasoning tokens rather than being forced to jump straight to the answer.',
    principles_zh: '当允许生成中间推理token而不是被迫直接跳转到答案时，大型语言模型在逻辑任务上的表现要好得多。',
    principles_ja: '大規模言語モデルは、答えに直接ジャンプするように強制されるのではなく、中間の推論トークンを生成することを許可されると、論理タスクで大幅に優れたパフォーマンスを発揮します。',
    architecture: '1. **Prompting**: Add "Let\'s think step by step" to the system prompt.\n2. **Generation**: Model outputs reasoning trace (e.g., "First I need to... then I will...").\n3. **Extraction**: The final answer is usually at the end of the generation.',
    architecture_zh: '1. **提示**：在系统提示中添加"让我们一步一步思考"。\n2. **生成**：模型输出推理轨迹。\n3. **提取**：最终答案通常在生成的末尾。',
    architecture_ja: '1. **プロンプト**: システムプロンプトに「ステップバイステップで考えましょう」を追加します。\n2. **生成**: モデルは推論トレースを出力します。\n3. **抽出**: 最終的な答えは通常、生成の最後にあります。',
    diagram: `graph LR
    Q[Question] --> Prompt["Let's think step by step"]
    Prompt --> Step1[Reasoning Step 1]
    Step1 --> Step2[Reasoning Step 2]
    Step2 --> Step3[Reasoning Step 3]
    Step3 --> Answer[Final Answer]`,
    codeExample: `
def cot_reasoning(problem):
    # Guide model to think
    cot_prompt = f"""
    Problem: {problem}
    Let's think step by step:
    1.
    """
    
    reasoning = llm.generate(cot_prompt)
    
    # Derive final answer from reasoning
    final_prompt = f"""
    Problem: {problem}
    Reasoning: {reasoning}
    Therefore, the final answer is:
    """
    
    answer = llm.generate(final_prompt)
    return answer
`
  },
  {
    id: 'self-reflection',
    name: 'Self-Reflection (Reflexion)',
    name_zh: '自我反思 (Reflexion)',
    name_ja: '自己反省 (Reflexion)',
    description: 'The agent critiques its own output and iteratively improves it before showing it to the user.',
    description_zh: '智能体在向用户展示之前批评自己的输出并迭代改进。',
    description_ja: 'エージェントは自身の出力を批評し、ユーザーに表示する前に反復的に改善します。',
    useCase: 'Content writing, Code generation, Quality control.',
    useCase_zh: '内容写作，代码生成，质量控制。',
    useCase_ja: 'コンテンツ執筆、コード生成、品質管理。',
    complexity: 'Medium',
    tags: ['Quality', 'Iterative', 'Optimization'],
    principles: 'Mimics human review processes. A "Critique" step is added after generation. The model is asked to find flaws in its own work, then a "Revision" step fixes those flaws.',
    principles_zh: '模仿人类审查过程。在生成后添加"批评"步骤。模型被要求找出自己工作中的缺陷，然后"修订"步骤修复这些缺陷。',
    principles_ja: '人間のレビュープロセスを模倣します。生成後に「批評」ステップが追加されます。モデルは自身の作業の欠陥を見つけるように求められ、その後「修正」ステップでそれらの欠陥が修正されます。',
    architecture: '1. **Draft**: Agent generates an initial response.\n2. **Critique**: Agent is prompted to "Critique the previous response for errors".\n3. **Refine**: Agent generates a new response based on the critique.\n4. **Repeat**: Loop until quality threshold is met.',
    architecture_zh: '1. **草稿**：智能体生成初始响应。\n2. **批评**：提示智能体"批评先前的响应中的错误"。\n3. **精炼**：智能体根据批评生成新响应。\n4. **重复**：循环直到满足质量阈值。',
    architecture_ja: '1. **ドラフト**: エージェントが初期応答を生成します。\n2. **批評**: エージェントは「前の応答のエラーを批評する」ように求められます。\n3. **改善**: エージェントは批評に基づいて新しい応答を生成します。\n4. **繰り返し**: 品質しきい値が満たされるまでループします。',
    diagram: `graph TD
    Task --> Draft
    Draft --> Critique
    Critique -->|Flaws Found| Refine
    Refine --> Draft
    Critique -->|Good Enough| Final[Final Output]`,
    codeExample: `
class SelfReflectionAgent:
    def generate_and_reflect(self, task):
        # 1. Initial Draft
        initial_response = llm.generate(f"Task: {task}")
        
        # 2. Self-Critique
        reflection_prompt = f"""
        Task: {task}
        Initial Response: {initial_response}
        Please critique this response and suggest improvements:
        """
        reflection = llm.generate(reflection_prompt)
        
        # 3. Improvement
        improvement_prompt = f"""
        Task: {task}
        Initial Response: {initial_response}
        Reflection: {reflection}
        Generate an improved response:
        """
        return llm.generate(improvement_prompt)
`
  },
  {
    id: 'memory-augmented',
    name: 'Memory-Augmented Agent',
    name_zh: '记忆增强智能体',
    name_ja: 'メモリ拡張エージェント',
    description: 'Agents equipped with long-term memory (Vector Database) to recall past interactions or specific knowledge over long periods.',
    description_zh: '配备长期记忆（向量数据库）的智能体，可以回忆过去的交互或特定知识。',
    description_ja: '長期記憶（ベクトルデータベース）を備えたエージェントで、過去の対話や特定の知識を長期間にわたって想起できます。',
    useCase: 'Personal Assistants, Long-term Learning, Relationship Management.',
    useCase_zh: '个人助理，长期学习，关系管理。',
    useCase_ja: 'パーソナルアシスタント、長期学習、関係管理。',
    complexity: 'High',
    tags: ['Memory', 'Personalization', 'Vector DB'],
    principles: 'LLMs have limited context windows. Memory augmentation offloads history to a database. Relevant memories are retrieved based on semantic similarity to the current context.',
    principles_zh: 'LLM的上下文窗口有限。记忆增强将历史记录卸载到数据库。根据与当前上下文的语义相似性检索相关记忆。',
    principles_ja: 'LLMにはコンテキストウィンドウに制限があります。メモリ拡張は履歴をデータベースにオフロードします。現在のコンテキストとの意味的類似性に基づいて、関連するメモリが取得されます。',
    architecture: '1. **Input**: Receive user message.\n2. **Recall**: Search Vector DB for relevant past conversations/facts.\n3. **Context**: Inject retrieved memories into the system prompt.\n4. **Generate**: Produce response.\n5. **Store**: Save the new interaction into the Vector DB.',
    architecture_zh: '1. **输入**：接收用户消息。\n2. **回忆**：在向量数据库中搜索相关的过去对话/事实。\n3. **上下文**：将检索到的记忆注入系统提示。\n4. **生成**：产生响应。\n5. **存储**：将新交互保存到向量数据库。',
    architecture_ja: '1. **入力**: ユーザーメッセージを受信します。\n2. **想起**: 関連する過去の会話/事実をベクトルDBで検索します。\n3. **コンテキスト**: 取得したメモリをシステムプロンプトに注入します。\n4. **生成**: 応答を生成します。\n5. **保存**: 新しい対話をベクトルDBに保存します。',
    diagram: `graph TD
    User --> Input
    Input --> Search[Vector Search]
    DB[(Long Term Memory)] -->|Results| Search
    Search --> Context
    Context --> LLM
    LLM --> Response
    Input -->|Save| DB
    Response -->|Save| DB`,
    codeExample: `
class MemoryAugmentedAgent:
    def __init__(self):
        self.short_term_memory = []
        self.long_term_memory = VectorDatabase()
    
    def process(self, input_text):
        # 1. Retrieve Memory
        relevant_memories = self.long_term_memory.search(input_text)
        
        # 2. Build Context
        context = {
            'current_input': input_text,
            'short_term': self.short_term_memory,
            'long_term': relevant_memories
        }
        
        # 3. Generate
        response = llm.generate_with_context(context)
        
        # 4. Store
        self.short_term_memory.append((input_text, response))
        self.long_term_memory.store(input_text, response)
        
        return response
`
  },
  {
    id: 'hierarchical',
    name: 'Hierarchical Agent',
    name_zh: '层级智能体',
    name_ja: '階層型エージェント',
    description: 'A multi-level structure where high-level agents (Managers) plan and delegate, while low-level agents (Executors) perform specific tasks.',
    description_zh: '一种多层结构，高层智能体（经理）负责计划和委派，而低层智能体（执行者）执行具体任务。',
    description_ja: '高レベルのエージェント（マネージャー）が計画と委任を行い、低レベルのエージェント（実行者）が特定のタスクを実行するマルチレベル構造。',
    useCase: 'Large Enterprise Systems, Complex Projects, Autonomous Companies.',
    useCase_zh: '大型企业系统，复杂项目，自主公司。',
    useCase_ja: '大企業システム、複雑なプロジェクト、自律型企業。',
    complexity: 'Very High',
    tags: ['Management', 'Scale', 'Delegation'],
    principles: ' mimics corporate hierarchy. The top-level agent does not know *how* to do the task, only *who* to ask. This abstraction allows for scaling to extremely complex tasks.',
    principles_zh: '模仿企业层级。顶层智能体不知道*如何*做任务，只知道*问谁*。这种抽象允许扩展到极其复杂的任务。',
    principles_ja: '企業の階層構造を模倣します。トップレベルのエージェントはタスクの実行方法を知らず、誰に頼むべきかだけを知っています。この抽象化により、非常に複雑なタスクへのスケーリングが可能になります。',
    architecture: '1. **Top Level**: Planner breaks down goal into sub-goals.\n2. **Middle Level**: Coordinators assign sub-goals to specific departments/workers.\n3. **Bottom Level**: Executors perform the work using specific tools.\n4. **Reporting**: Results bubble up the hierarchy, being summarized at each level.',
    architecture_zh: '1. **顶层**：规划者将目标分解为子目标。\n2. **中层**：协调员将子目标分配给特定部门/工人。\n3. **底层**：执行者使用特定工具执行工作。\n4. **汇报**：结果在层级中上报，在每一级进行汇总。',
    architecture_ja: '1. **トップレベル**: プランナーが目標をサブ目標に分解します。\n2. **中間レベル**: コーディネーターがサブ目標を特定の部門/ワーカーに割り当てます。\n3. **ボトムレベル**: 実行者が特定のツールを使用して作業を実行します。\n4. **報告**: 結果は階層を上がり、各レベルで要約されます。',
    diagram: `graph TD
    CEO[High Level Planner] -->|Goal| Manager1[Coordinator A]
    CEO -->|Goal| Manager2[Coordinator B]
    Manager1 -->|Task| Worker1[Executor]
    Manager1 -->|Task| Worker2[Executor]
    Manager2 -->|Task| Worker3[Executor]
    Worker1 -->|Result| Manager1
    Manager1 -->|Summary| CEO`,
    codeExample: `
class HierarchicalAgent:
    def __init__(self):
        self.planner = HighLevelPlanner()
        self.coordinators = [TaskCoordinator() for _ in range(3)]
        self.executors = [BasicExecutor() for _ in range(10)]
    
    def execute_task(self, goal):
        # 1. Plan
        plan = self.planner.create_plan(goal)
        
        # 2. Coordinate
        subtasks = []
        for coordinator in self.coordinators:
            subtask = coordinator.assign_task(plan)
            subtasks.append(subtask)
        
        # 3. Execute
        results = []
        for subtask in subtasks:
            executor = self.select_available_executor()
            result = executor.execute(subtask)
            results.append(result)
        
        return self.integrate_results(results)
`
  },
  {
    id: 'swarm',
    name: 'Swarm Intelligence',
    name_zh: '群体智能',
    name_ja: '群知能',
    description: 'Many simple agents follow simple local rules to exhibit complex global behavior, similar to ants or bees.',
    description_zh: '许多简单的智能体遵循简单的局部规则，表现出复杂的全局行为，类似于蚂蚁或蜜蜂。',
    description_ja: '多くのアリやハチのように、多くの単純なエージェントが単純なローカルルールに従って、複雑なグローバルな動作を示します。',
    useCase: 'Distributed Optimization, Market Prediction, Simulation.',
    useCase_zh: '分布式优化，市场预测，仿真。',
    useCase_ja: '分散最適化、市場予測、シミュレーション。',
    complexity: 'High',
    tags: ['Emergence', 'Distributed', 'Robustness'],
    principles: 'No central controller. Agents interact with their neighbors and the environment. The "intelligence" emerges from the collective, not the individual.',
    principles_zh: '无中央控制器。智能体与邻居和环境交互。"智能"从集体中涌现，而非个体。',
    principles_ja: '中央コントローラーはありません。エージェントは近隣のエージェントや環境と相互作用します。「知能」は個体ではなく集団から出現します。',
    architecture: '1. **Population**: Initialize N simple agents.\n2. **Interaction**: Agents share data (e.g., "best location found so far") with neighbors.\n3. **Update**: Agents adjust their state/position based on neighbor data.\n4. **Convergence**: The system converges on an optimal solution over time.',
    architecture_zh: '1. **种群**：初始化N个简单智能体。\n2. **交互**：智能体与邻居共享数据（如"目前发现的最佳位置"）。\n3. **更新**：智能体根据邻居数据调整状态/位置。\n4. **收敛**：系统随时间收敛到最优解。',
    architecture_ja: '1. **個体群**: N個の単純なエージェントを初期化します。\n2. **相互作用**: エージェントは近隣のエージェントとデータ（例：「これまでに発見された最適な場所」）を共有します。\n3. **更新**: エージェントは近隣のデータに基づいて状態/位置を調整します。\n4. **収束**: システムは時間の経過とともに最適なソリューションに収束します。',
    diagram: `graph TD
    A[Agent A] <--> B[Agent B]
    B <--> C[Agent C]
    C <--> D[Agent D]
    D <--> A
    A <--> C
    Note[Global Behavior Emerges from Local Links]`,
    codeExample: `
class SwarmAgent:
    def update(self, neighbors, global_best):
        # 1. Update personal best
        if self.fitness() > self.personal_best_fitness:
            self.personal_best = self.position
        
        # 2. Collaborate with neighbors
        neighbors_best = max([n.personal_best for n in neighbors])
        
        # 3. Move/Update State
        self.velocity = self.update_velocity(neighbors_best, global_best)
        self.position = self.position + self.velocity
    
    def swarm_optimization(self, iterations):
        global_best = None
        for _ in range(iterations):
            for agent in self.swarm:
                neighbors = self.get_neighbors(agent)
                agent.update(neighbors, global_best)
            global_best = self.get_global_best()
        return global_best
`
  },
  {
    id: 'state-machine',
    name: 'State Machine Agent',
    name_zh: '状态机智能体',
    name_ja: 'ステートマシンエージェント',
    description: 'Combines explicit state management with function calling. The agent transitions between predefined states (e.g., "Analysis" -> "Execution") based on rules.',
    description_zh: '结合显式状态管理和函数调用。智能体基于规则在预定义状态（如"分析" -> "执行"）之间转换。',
    description_ja: '明示的な状態管理と関数呼び出しを組み合わせます。エージェントは、ルールに基づいて事前定義された状態（例：「分析」->「実行」）間を遷移します。',
    useCase: 'Business Processes, Customer Support Flows, Order Fulfillment.',
    useCase_zh: '业务流程，客户支持流，订单履行。',
    useCase_ja: 'ビジネスプロセス、カスタマーサポートフロー、注文処理。',
    complexity: 'Medium',
    tags: ['FSM', 'Control Flow', 'Deterministic'],
    principles: 'Constrains the Agent\'s freedom to ensure reliability. In each state, the agent has a different System Prompt and different available Tools.',
    principles_zh: '限制智能体的自由度以确保可靠性。在每个状态下，智能体拥有不同的系统提示和不同的可用工具。',
    principles_ja: '信頼性を確保するためにエージェントの自由を制限します。各状態で、エージェントは異なるシステムプロンプトと利用可能なツールを持ちます。',
    architecture: '1. **State Definition**: Define states (e.g., INITIAL, PLAN, ACT, REVIEW).\n2. **Transition Logic**: Define rules for moving between states (e.g., "If tool success, go to REVIEW").\n3. **Execution**: Loop runs logic specific to current state.\n4. **Output**: Final result is produced only in the Terminal state.',
    architecture_zh: '1. **状态定义**：定义状态（如初始、计划、行动、审查）。\n2. **转换逻辑**：定义状态间移动的规则。\n3. **执行**：循环运行特定于当前状态的逻辑。\n4. **输出**：仅在终止状态产生最终结果。',
    architecture_ja: '1. **状態定義**: 状態（例：INITIAL、PLAN、ACT、REVIEW）を定義します。\n2. **遷移ロジック**: 状態間を移動するためのルール（例：「ツールが成功したらREVIEWに移動」）を定義します。\n3. **実行**: ループは現在の状態に固有のロジックを実行します。\n4. **出力**: 最終結果はターミナル状態でのみ生成されます。',
    diagram: `stateDiagram-v2
    [*] --> Initial
    Initial --> Analysis: Input Received
    Analysis --> Decision: Analysis Complete
    Decision --> Execution: Plan Approved
    Execution --> Completion: Tools Success
    Execution --> Analysis: Error
    Completion --> [*]`,
    codeExample: `
class StateMachineAgent:
    def run(self, user_input):
        # 1. Execute State Logic
        if self.state == "PROCESSING":
            return self._process_input(user_input)
        elif self.state == "ANALYSIS":
            return self._analyze_context()
        elif self.state == "EXECUTION":
            return self._execute_actions()
            
        # 2. Call LLM
        response = llm.generate(self.messages)
        self.messages.append(response)
        
        # 3. Check Transition Tokens
        if "[ANALYSIS_COMPLETE]" in response:
            self.state = "DECISION"
            return self.run() # Recursion for next state
            
        return response
`
  }
];

export const CODING_TOOLS: CodingTool[] = [
  {
    id: 'cursor',
    name: 'Cursor',
    type: 'IDE',
    coreMechanism: 'Shadow Workspace & RAG',
    coreMechanism_zh: '影子工作区 & RAG',
    coreMechanism_ja: 'シャドウワークスペース & RAG',
    relatedPatternId: 'shadow-workspace',
    description: 'A VS Code fork that integrates "Composer" (Agent) and "Tab" (FIM). It uses a shadow workspace to generate code diffs and RAG to index the entire codebase for context-aware answers.',
    description_zh: 'VS Code的分支，集成了"Composer"（智能体）和"Tab"（FIM）。它使用影子工作区生成代码差异，并使用RAG索引整个代码库以提供上下文感知的答案。',
    description_ja: '"Composer"（エージェント）と"Tab"（FIM）を統合したVS Codeフォーク。シャドウワークスペースを使用してコードの差分を生成し、RAGを使用してコードベース全体にインデックスを付け、コンテキストを認識した回答を提供します。',
    features: ['Shadow Workspace (Composer)', 'Local/Remote Codebase Indexing', 'Fill-In-Middle Autocomplete'],
    features_zh: ['影子工作区 (Composer)', '本地/远程代码库索引', '中间填充自动补全'],
    features_ja: ['シャドウワークスペース (Composer)', 'ローカル/リモートコードベースインデックス', 'Fill-In-Middle 自動補完'],
    website: 'https://cursor.com'
  },
  {
    id: 'windsurf',
    name: 'Windsurf (Codeium)',
    type: 'IDE',
    coreMechanism: 'Deep Context Flows (Cascades)',
    coreMechanism_zh: '深度上下文流 (Cascades)',
    coreMechanism_ja: 'ディープコンテキストフロー (Cascades)',
    relatedPatternId: 'explicit-function',
    description: 'Built by Codeium. Features "Flows" which allow the agent to run terminal commands, read output, and edit files iteratively. It maintains a deep understanding of user intent via "Cascades".',
    description_zh: '由Codeium构建。功能"Flows"允许智能体运行终端命令，读取输出并迭代编辑文件。它通过"Cascades"保持对用户意图的深刻理解。',
    description_ja: 'Codeiumによって構築されました。「Flows」機能により、エージェントはターミナルコマンドを実行し、出力を読み取り、ファイルを反復的に編集できます。「Cascades」を通じてユーザーの意図を深く理解します。',
    features: ['Agentic Terminal Access', 'Deep Context Awareness', 'Iterative Refactoring'],
    features_zh: ['智能体终端访问', '深度上下文感知', '迭代重构'],
    features_ja: ['エージェント端末アクセス', '深いコンテキスト認識', '反復的リファクタリング'],
    website: 'https://codeium.com/windsurf'
  },
  {
    id: 'trae',
    name: 'Trae',
    type: 'IDE',
    coreMechanism: 'Data-Driven Context & Auto-Apply',
    coreMechanism_zh: '数据驱动上下文 & 自动应用',
    coreMechanism_ja: 'データ駆動型コンテキスト & 自動適用',
    relatedPatternId: 'rag-agent',
    description: 'ByteDance\'s AI IDE. Similar to Cursor/Windsurf, it focuses on high-speed code generation and intelligent context gathering from the project structure.',
    description_zh: '字节跳动的AI IDE。类似于Cursor/Windsurf，它专注于高速代码生成和从项目结构中智能收集上下文。',
    description_ja: 'ByteDanceのAI IDE。Cursor/Windsurfと同様に、高速なコード生成とプロジェクト構造からのインテリジェントなコンテキスト収集に重点を置いています。',
    features: ['Intelligent Context', 'Chat-to-Code', 'Codebase Visualization'],
    features_zh: ['智能上下文', '对话转代码', '代码库可视化'],
    features_ja: ['インテリジェントコンテキスト', 'チャットからコードへ', 'コードベースの可視化'],
    website: 'https://www.trae.ai'
  },
  {
    id: 'zed',
    name: 'Zed',
    type: 'IDE',
    coreMechanism: 'CRDT & GPU Rendering',
    coreMechanism_zh: 'CRDT & GPU 渲染',
    coreMechanism_ja: 'CRDT & GPU レンダリング',
    relatedPatternId: 'native-crdt',
    description: 'High-performance Rust-based editor. Uses CRDTs natively for collaboration, treating the AI as a collaborator in the document rather than a plugin wrapper.',
    description_zh: '高性能Rust基编辑器。原生使用CRDT进行协作，将AI视为文档中的协作者，而不是插件包装器。',
    description_ja: '高性能なRustベースのエディタ。コラボレーションにCRDTをネイティブに使用し、AIをプラグインラッパーではなくドキュメント内のコラボレーターとして扱います。',
    features: ['Native CRDT', 'GPU Accelerated', 'Model Agnostic Chat'],
    features_zh: ['原生 CRDT', 'GPU 加速', '模型无关聊天'],
    features_ja: ['ネイティブ CRDT', 'GPU アクセラレーション', 'モデルに依存しないチャット'],
    website: 'https://zed.dev'
  },
  {
    id: 'aider',
    name: 'Aider',
    type: 'CLI',
    coreMechanism: 'AST-based "Repo Map"',
    coreMechanism_zh: '基于AST的"仓库地图"',
    coreMechanism_ja: 'ASTベースの「リポジトリマップ」',
    relatedPatternId: 'ast-semantic',
    description: 'A CLI tool that works with your local git repo. It generates a "Repo Map" (a compressed AST summary of your code) to fit the entire project structure into the LLM context window.',
    description_zh: '一个与本地git仓库配合使用的CLI工具。它生成"仓库地图"（代码的压缩AST摘要），以将整个项目结构放入LLM上下文窗口中。',
    description_ja: 'ローカルのgitリポジトリと連携するCLIツール。「リポジトリマップ」（コードの圧縮されたASTサマリー）を生成し、プロジェクト構造全体をLLMコンテキストウィンドウに収めます。',
    features: ['Git Integration', 'AST Repo Map', 'Terminal UI'],
    features_zh: ['Git 集成', 'AST 仓库地图', '终端 UI'],
    features_ja: ['Git 統合', 'AST リポジトリマップ', 'ターミナル UI'],
    website: 'https://aider.chat'
  },
  {
    id: 'codebuddy',
    name: 'CodeBuddy',
    type: 'Extension',
    coreMechanism: 'Sandbox Execution',
    coreMechanism_zh: '沙盒执行',
    coreMechanism_ja: 'サンドボックス実行',
    relatedPatternId: 'shadow-workspace',
    description: 'An agent that can not only write code but execute it in a sandbox to verify functionality before committing changes.',
    description_zh: '一个不仅能写代码，还能在提交更改前在沙盒中执行代码以验证功能的智能体。',
    description_ja: 'コードを書くだけでなく、変更をコミットする前にサンドボックスで実行して機能を検証できるエージェント。',
    features: ['Sandboxed Execution', 'Self-Correction', 'Web Search'],
    features_zh: ['沙盒执行', '自我修正', '网络搜索'],
    features_ja: ['サンドボックス実行', '自己修正', 'ウェブ検索'],
    website: 'https://codebuddy.ca'
  },
  {
    id: 'kiro',
    name: 'Kiro',
    type: 'IDE',
    coreMechanism: 'Agentic Workflow',
    coreMechanism_zh: '智能体工作流',
    coreMechanism_ja: 'エージェンティックワークフロー',
    relatedPatternId: 'multi-agent',
    description: 'A newer entrant focusing on autonomous agent workflows directly within the editor, automating repetitive refactoring tasks.',
    description_zh: '专注于编辑器内自主智能体工作流的新进者，自动化重复的重构任务。',
    description_ja: 'エディタ内で直接自律型エージェントワークフローに焦点を当て、反復的なリファクタリングタスクを自動化する新規参入者。',
    features: ['Task Automation', 'Agentic Refactoring'],
    features_zh: ['任务自动化', '智能体重构'],
    features_ja: ['タスク自動化', 'エージェンティックリファクタリング'],
    website: 'https://kiro.ai'
  },
  {
    id: 'qoder',
    name: 'Qoder',
    type: 'Platform',
    coreMechanism: 'End-to-End Testing Agent',
    coreMechanism_zh: '端到端测试智能体',
    coreMechanism_ja: 'エンドツーエンドテストエージェント',
    relatedPatternId: 'shadow-workspace',
    description: 'Focuses on ensuring code quality by generating and running tests (Shadow Workspace pattern) to validate generated code.',
    description_zh: '专注于通过生成和运行测试（影子工作区模式）来验证生成的代码，从而确保代码质量。',
    description_ja: '生成されたコードを検証するためにテストを生成して実行する（シャドウワークスペースパターン）ことで、コード品質を確保することに重点を置いています。',
    features: ['Test-Driven Generation', 'Quality Assurance'],
    features_zh: ['测试驱动生成', '质量保证'],
    features_ja: ['テスト駆動生成', '品質保証'],
    website: 'https://qoder.ai'
  },
  {
    id: 'antigravity',
    name: 'Antigravity',
    type: 'IDE',
    coreMechanism: 'Visual Debugging & Object Tracking',
    coreMechanism_zh: '可视化调试 & 对象跟踪',
    coreMechanism_ja: 'ビジュアルデバッグ & オブジェクト追跡',
    relatedPatternId: 'structured-text',
    description: 'Python-focused IDE that visualizes code execution flow. The AI uses this runtime data to understand bugs better than static analysis.',
    description_zh: '专注于Python的IDE，可视化代码执行流。AI使用此时运行时数据比静态分析更好地理解Bug。',
    description_ja: 'コード実行フローを可視化するPython重視のIDE。AIはこのランタイムデータを使用して、静的分析よりもバグをよりよく理解します。',
    features: ['Time-Travel Debugging', 'Visual Flow', 'Runtime Analysis'],
    features_zh: ['时间旅行调试', '可视化流程', '运行时分析'],
    features_ja: ['タイムトラベルデバッグ', 'ビジュアルフロー', 'ランタイム分析'],
    website: 'https://antigravity.ai'
  },
  {
    id: 'codex-cli',
    name: 'OpenAI Codex / GitHub Copilot CLI',
    type: 'CLI',
    coreMechanism: 'Command Translation',
    coreMechanism_zh: '命令翻译',
    coreMechanism_ja: 'コマンド翻訳',
    relatedPatternId: 'structured-text',
    description: 'Translates natural language into shell commands (Bash, PowerShell, Zsh). Uses specialized fine-tuning for shell syntax.',
    description_zh: '将自然语言翻译为Shell命令（Bash, PowerShell, Zsh）。针对Shell语法使用了专门的微调。',
    description_ja: '自然言語をシェルコマンド（Bash、PowerShell、Zsh）に翻訳します。シェル構文用に特化したファインチューニングを使用します。',
    features: ['Natural Language to Bash', 'Command Explanation'],
    features_zh: ['自然语言转Bash', '命令解释'],
    features_ja: ['自然言語からBashへ', 'コマンド解説'],
    website: 'https://githubnext.com/projects/copilot-cli'
  },
  {
    id: 'cloud-code',
    name: 'Google Cloud Code',
    type: 'Extension',
    coreMechanism: 'Infrastructure aware (Gemini)',
    coreMechanism_zh: '基础设施感知 (Gemini)',
    coreMechanism_ja: 'インフラストラクチャ認識 (Gemini)',
    relatedPatternId: 'explicit-function',
    description: 'IDE extension for VS Code/IntelliJ. Uses Gemini to understand Kubernetes manifests, Terraform, and Cloud deployment configurations.',
    description_zh: 'VS Code/IntelliJ的IDE扩展。使用Gemini理解Kubernetes清单、Terraform和云部署配置。',
    description_ja: 'VS Code/IntelliJ用のIDE拡張機能。Geminiを使用して、Kubernetesマニフェスト、Terraform、およびクラウド展開構成を理解します。',
    features: ['Kubernetes YAML Gen', 'Cloud Debugging', 'Duet AI Integration'],
    features_zh: ['K8s YAML 生成', '云调试', 'Duet AI 集成'],
    features_ja: ['K8s YAML 生成', 'クラウドデバッグ', 'Duet AI 統合'],
    website: 'https://cloud.google.com/code'
  }
];

export const BUILD_EXAMPLES: BuilderExample[] = [
  {
    id: 'simple-agent',
    title: 'Minimal General Agent',
    title_zh: '极简通用智能体',
    title_ja: '最小限の汎用エージェント',
    description: 'A 30-line Python script that creates a chatbot capable of checking weather and time using Function Calling.',
    description_zh: '一个30行的Python脚本，创建一个能够使用函数调用检查天气和时间的聊天机器人。',
    description_ja: 'Function Callingを使用して天気と時刻を確認できるチャットボットを作成する30行のPythonスクリプト。',
    language: 'Python',
    difficulty: 'Beginner',
    explanation: 'This example uses standard JSON-based tool definitions. It defines a `tools` list and a loop that checks `if tool_calls` exists in the response. If it does, it runs the function and sends the result back.',
    explanation_zh: '此示例使用标准的基于JSON的工具定义。它定义了一个`tools`列表和一个循环，检查响应中是否存在`tool_calls`。如果存在，它运行该函数并将结果发回。',
    explanation_ja: 'この例では、標準のJSONベースのツール定義を使用しています。`tools`リストと、レスポンスに`tool_calls`が存在するかどうかを確認するループを定義します。存在する場合、関数を実行して結果を送り返します。',
    code: `import os
from google import genai
from google.genai import types

# 1. Setup Client
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# 2. Define Tools (The actual functions)
def get_weather(city: str):
    """Returns fake weather data."""
    return f"The weather in {city} is Sunny and 75F."

def get_time(timezone: str):
    """Returns fake time data."""
    return f"The time in {timezone} is 2:00 PM."

# 3. Create the Tool Configuration
# We wrap the functions so the SDK can inspect them
tools_config = [get_weather, get_time]

# 4. Chat Loop
chat = client.chats.create(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        tools=tools_config,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(
            disable=False # SDK handles the loop automatically!
        )
    )
)

print("Agent Ready. Type 'quit' to exit.")
while True:
    user_input = input("User: ")
    if user_input.lower() == "quit": break
    
    # The SDK executes the function and sends the result back automatically
    response = chat.send_message(user_input)
    print(f"Agent: {response.text}")`
  },
  {
    id: 'coding-agent',
    title: 'Minimal Coding Agent',
    title_zh: '极简编程智能体',
    title_ja: '最小限のコーディングエージェント',
    description: 'An agent that can read files, write code, and execute commands to fix bugs. This is the core logic behind tools like Aider or OpenDevin.',
    description_zh: '一个可以读取文件、编写代码并执行命令来修复Bug的智能体。这是Aider或OpenDevin等工具背后的核心逻辑。',
    description_ja: 'ファイルを読み取り、コードを記述し、コマンドを実行してバグを修正できるエージェント。これは、AiderやOpenDevinなどのツールの背後にあるコアロジックです。',
    language: 'Python',
    difficulty: 'Intermediate',
    explanation: 'This agent has "hands". We give it `read_file`, `write_file`, and `run_shell` tools. The system instruction tells it to act like a developer.',
    explanation_zh: '这个智能体有"手"。我们给它`read_file`、`write_file`和`run_shell`工具。系统指令告诉它像开发者一样行事。',
    explanation_ja: 'このエージェントには「手」があります。`read_file`、`write_file`、および`run_shell`ツールを与えます。システム命令は、開発者のように振る舞うように指示します。',
    code: `import os
import subprocess
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# --- The "Hands" of the Agent ---

def read_file(filepath: str):
    """Reads content of a file."""
    try:
        with open(filepath, 'r') as f: return f.read()
    except Exception as e: return str(e)

def write_file(filepath: str, content: str):
    """Writes content to a file."""
    try:
        with open(filepath, 'w') as f: f.write(content)
        return "Success"
    except Exception as e: return str(e)

def run_shell(command: str):
    """Executes a shell command and returns stdout/stderr."""
    # WARNING: Dangerous in production without sandbox
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return f"STDOUT: {result.stdout}\\nSTDERR: {result.stderr}"

# --- Agent Setup ---

sys_prompt = """
You are a Coding Agent. 
1. Always read the file first before editing.
2. Use write_file to apply changes.
3. Use run_shell to verify your code works.
"""

chat = client.chats.create(
    model="gemini-2.0-flash",
    config=types.GenerateContentConfig(
        system_instruction=sys_prompt,
        tools=[read_file, write_file, run_shell],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False)
    )
)

# --- Interactive Session ---
print("Coding Agent Initialized. (pwd: " + os.getcwd() + ")")
while True:
    req = input("Task > ")
    if req == "exit": break
    res = chat.send_message(req)
    print(f"Agent > {res.text}")
`
  },
  {
    id: 'langgraph-teacher',
    title: 'LangGraph Multi-Teacher Agent',
    title_zh: 'LangGraph 多教师智能体',
    title_ja: 'LangGraph マルチ教師エージェント',
    description: 'A stateful Routing Agent that acts as a Supervisor, directing your questions to specialized sub-agents (Coder, English Teacher, Logic Teacher, Doctor).',
    description_zh: '一个有状态的路由智能体，充当监督者，将你的问题引导给专门的子智能体（编码员、英语老师、逻辑老师、医生）。',
    description_ja: 'スーパーバイザーとして機能するステートフルなルーティングエージェント。質問を専門のサブエージェント（コーダー、英語教師、論理教師、医師）に送信します。',
    language: 'Python',
    difficulty: 'Intermediate',
    explanation: 'This uses LangGraph\'s `StateGraph`. We define a "Router" node that classifies the user intention (e.g., "This is a coding question"). Based on that class, the graph transitions to the specific "Teacher" node. Each teacher has a unique System Prompt.',
    explanation_zh: '这使用了LangGraph的`StateGraph`。我们定义了一个"路由"节点来分类用户意图（例如，"这是一个编程问题"）。基于该分类，图转换到特定的"教师"节点。每个教师都有独特的系统提示。',
    explanation_ja: 'これはLangGraphの`StateGraph`を使用します。ユーザーの意図を分類する「ルーター」ノードを定義します（例：「これはコーディングの質問です」）。そのクラスに基づいて、グラフは特定の「教師」ノードに遷移します。各教師には独自のシステムプロンプトがあります。',
    code: `import os
from typing import Annotated, Literal, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# 1. Setup LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=os.environ["GEMINI_API_KEY"])

# 2. Define State (Shared memory for the graph)
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 3. Define Teacher Nodes
def create_teacher_node(role: str, prompt: str):
    """Factory to create specialized teacher functions."""
    def teacher_node(state: AgentState):
        # Combine system prompt with conversation history
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}
    return teacher_node

# Instantiate specialized nodes
programming_node = create_teacher_node("coder", "You are a Programming Teacher. Explain concepts with code examples.")
english_node = create_teacher_node("english", "You are an English Teacher. Correct grammar and explain vocabulary.")
logic_node = create_teacher_node("logic", "You are a Logic Teacher. Analyze arguments and point out fallacies.")
doctor_node = create_teacher_node("doctor", "You are a Family Doctor. Provide general health advice but always disclaim you are an AI.")

# 4. Define Router (The "Supervisor")
def router_node(state: AgentState) -> Literal["coder", "english", "logic", "doctor"]:
    last_msg = state["messages"][-1].content
    # Ask LLM to classify intent
    router_prompt = [
        SystemMessage(content="Classify the user input into one of these categories: 'coder', 'english', 'logic', 'doctor'. Return ONLY the category name."),
        HumanMessage(content=last_msg)
    ]
    category = llm.invoke(router_prompt).content.strip().lower()
    
    # Fallback safety
    valid_cats = ["coder", "english", "logic", "doctor"]
    if category not in valid_cats:
        return "logic" # Default fallback
    return category

# 5. Build Graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("coder", programming_node)
workflow.add_node("english", english_node)
workflow.add_node("logic", logic_node)
workflow.add_node("doctor", doctor_node)

# Add conditional entry point (Router)
# The graph starts by calling 'router_node', which returns the name of the next node to visit.
workflow.set_conditional_entry_point(
    router_node,
    {
        "coder": "coder",
        "english": "english",
        "logic": "logic",
        "doctor": "doctor"
    }
)

# All nodes finish after answering
workflow.add_edge("coder", END)
workflow.add_edge("english", END)
workflow.add_edge("logic", END)
workflow.add_edge("doctor", END)

app = workflow.compile()

# 6. Run (Simulation)
print("Teacher Agent Router Ready...")
# In a real app, use a loop. Here is one interaction:
user_in = "How do I write a for loop in Python?"
print(f"Student: {user_in}")

# Stream the output
for event in app.stream({"messages": [HumanMessage(content=user_in)]}):
    for node_name, value in event.items():
        print(f"--- Node: {node_name} ---")
        print(value['messages'][-1].content)
`
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
