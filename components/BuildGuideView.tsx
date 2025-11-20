
import React, { useState } from 'react';
import { BuilderExample } from '../types';
import { Code2, Terminal, Check, Copy, Rocket, BookOpen } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface BuildGuideViewProps {
  examples: BuilderExample[];
  ui: any;
}

const BuildGuideView: React.FC<BuildGuideViewProps> = ({ examples, ui }) => {
  const [activeTab, setActiveTab] = useState(examples[0].id);

  const activeExample = examples.find(e => e.id === activeTab) || examples[0];

  const copyToClipboard = (code: string) => {
    navigator.clipboard.writeText(code);
  };

  return (
    <div className="animate-in fade-in duration-500">
        <div className="mb-10">
            <h2 className="text-3xl font-bold text-white mb-3">{ui.navBuild}</h2>
            <p className="text-slate-400">
               {ui.headerSubtitle}
            </p>
        </div>

        <div className="flex flex-col lg:flex-row gap-8">
            
            {/* Sidebar / Tabs */}
            <div className="lg:w-1/3 space-y-4">
                {examples.map(example => (
                    <button
                        key={example.id}
                        onClick={() => setActiveTab(example.id)}
                        className={`w-full text-left p-5 rounded-xl border transition-all duration-300 group ${
                            activeTab === example.id 
                            ? 'bg-indigo-900/20 border-indigo-500/50 shadow-lg shadow-indigo-900/20' 
                            : 'bg-slate-900/50 border-slate-800 hover:border-slate-600'
                        }`}
                    >
                        <div className="flex items-center justify-between mb-2">
                            <span className={`px-2 py-0.5 rounded text-xs font-bold uppercase ${
                                example.id === 'simple-agent' ? 'bg-green-500/10 text-green-400' : 'bg-blue-500/10 text-blue-400'
                            }`}>
                                {example.difficulty}
                            </span>
                            <div className={`p-2 rounded-lg ${activeTab === example.id ? 'bg-indigo-500/20 text-indigo-300' : 'bg-slate-800 text-slate-400'}`}>
                                {example.id === 'simple-agent' ? <Rocket size={18} /> : <Terminal size={18} />}
                            </div>
                        </div>
                        <h3 className={`text-lg font-bold mb-1 ${activeTab === example.id ? 'text-white' : 'text-slate-300'}`}>
                            {example.title}
                        </h3>
                        <p className="text-sm text-slate-400 leading-relaxed line-clamp-3">
                            {example.description}
                        </p>
                    </button>
                ))}

                <div className="p-5 bg-slate-900/80 border border-slate-800 rounded-xl">
                    <h4 className="flex items-center gap-2 font-semibold text-white mb-3">
                        <BookOpen size={18} className="text-purple-400"/>
                        {ui.whyWorks}
                    </h4>
                    <p className="text-sm text-slate-400 leading-relaxed">
                        Most "Agent Frameworks" are just wrappers around these exact API calls. 
                        By using the SDK's <code className="text-indigo-300">automatic_function_calling</code> feature, 
                        you get a production-grade agent loop (ReAct) without writing the recursion logic yourself.
                    </p>
                </div>
            </div>

            {/* Code Area */}
            <div className="lg:w-2/3">
                <div className="bg-slate-950 border border-slate-800 rounded-2xl overflow-hidden shadow-2xl">
                    <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/50">
                        <div className="flex items-center gap-3">
                            <div className="flex gap-1.5">
                                <div className="w-3 h-3 rounded-full bg-red-500/50"></div>
                                <div className="w-3 h-3 rounded-full bg-yellow-500/50"></div>
                                <div className="w-3 h-3 rounded-full bg-green-500/50"></div>
                            </div>
                            <span className="ml-2 text-sm font-mono text-slate-400">agent.py</span>
                        </div>
                        <button 
                            onClick={() => copyToClipboard(activeExample.code)}
                            className="flex items-center gap-2 text-xs font-medium text-slate-400 hover:text-white transition-colors bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded-lg"
                        >
                            <Copy size={14} />
                            {ui.copyCode}
                        </button>
                    </div>
                    
                    <div className="p-6 overflow-x-auto">
                         {/* Explanation Block */}
                        <div className="mb-6 p-4 bg-indigo-900/10 border border-indigo-500/20 rounded-xl">
                            <h4 className="text-indigo-300 font-semibold mb-1 flex items-center gap-2">
                                <Code2 size={16}/> {ui.logicBreakdown}
                            </h4>
                            <p className="text-sm text-slate-300">
                                {activeExample.explanation}
                            </p>
                        </div>

                        <div className="prose prose-invert max-w-none">
                            <ReactMarkdown
                                components={{
                                    code({node, className, children, ...props}) {
                                        return (
                                            <code className={`${className || ''} text-sm font-mono text-blue-200`} {...props}>
                                                {children}
                                            </code>
                                        )
                                    },
                                    pre({node, children, ...props}) {
                                        return (
                                            <pre className="bg-transparent p-0 m-0" {...props}>
                                                {children}
                                            </pre>
                                        )
                                    }
                                }}
                            >
                                {`\`\`\`python\n${activeExample.code}\n\`\`\``}
                            </ReactMarkdown>
                        </div>
                    </div>
                </div>
            </div>

        </div>
    </div>
  );
};

export default BuildGuideView;
