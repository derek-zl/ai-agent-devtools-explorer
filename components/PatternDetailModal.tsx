
import React from 'react';
import { Pattern } from '../types';
import { X, Network, BookOpen, Layers, Code, Copy } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface PatternDetailModalProps {
  pattern: Pattern | null;
  onClose: () => void;
  ui?: any;
}

const PatternDetailModal: React.FC<PatternDetailModalProps> = ({ pattern, onClose, ui }) => {
  if (!pattern) return null;

  // Helper to encode string for mermaid.ink
  const getSimpleMermaidUrl = (graph: string) => {
      const state = {
      code: graph, 
      mermaid: { theme: 'dark' }
    }
    const encoded = btoa(JSON.stringify(state));
    return `https://mermaid.ink/img/${encoded}`;
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div 
        className="bg-slate-900 border border-slate-700 w-full max-w-5xl max-h-[90vh] rounded-2xl shadow-2xl overflow-hidden flex flex-col animate-in zoom-in-95 duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-6 border-b border-slate-800 flex justify-between items-start bg-slate-900">
          <div>
            <h2 className="text-2xl font-bold text-white mb-1">{pattern.name}</h2>
            <div className="flex gap-2">
               {pattern.tags.map(tag => (
                 <span key={tag} className="text-xs font-mono text-indigo-400 bg-indigo-900/20 px-2 py-1 rounded">
                   {tag.toUpperCase()}
                 </span>
               ))}
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-2 hover:bg-slate-800 rounded-full text-slate-400 hover:text-white transition-colors"
          >
            <X size={24} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 bg-slate-950 space-y-8">
          
          {/* Diagram */}
          <div className="w-full bg-slate-900 border border-slate-800 rounded-xl p-4 flex justify-center items-center overflow-hidden min-h-[200px]">
             <img 
                src={getSimpleMermaidUrl(pattern.diagram)} 
                alt="Architecture Diagram"
                className="max-w-full h-auto"
             />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Principles */}
            <div>
                <h3 className="flex items-center gap-2 text-lg font-semibold text-indigo-300 mb-4">
                    <BookOpen size={20} />
                    {ui?.corePrinciples || 'Core Principles'}
                </h3>
                <div className="prose prose-invert prose-sm text-slate-300">
                    <ReactMarkdown>{pattern.principles}</ReactMarkdown>
                </div>
            </div>

            {/* Architecture */}
            <div>
                <h3 className="flex items-center gap-2 text-lg font-semibold text-indigo-300 mb-4">
                    <Layers size={20} />
                    {ui?.techArch || 'Technical Architecture'}
                </h3>
                <div className="prose prose-invert prose-sm text-slate-300">
                    <ReactMarkdown>{pattern.architecture}</ReactMarkdown>
                </div>
            </div>
          </div>

          {/* Implementation Code Example */}
          {pattern.codeExample && (
            <div className="w-full">
                <h3 className="flex items-center gap-2 text-lg font-semibold text-indigo-300 mb-4">
                    <Code size={20} />
                    {ui?.implRef || 'Implementation Reference'}
                </h3>
                <div className="relative bg-slate-900 rounded-xl border border-slate-800 overflow-hidden group">
                   <div className="absolute top-3 right-3 opacity-0 group-hover:opacity-100 transition-opacity">
                      <span className="text-xs text-slate-500 flex items-center gap-1 bg-slate-800 px-2 py-1 rounded">
                         <Copy size={12} /> Read-only
                      </span>
                   </div>
                   <div className="p-4 overflow-x-auto text-sm font-mono leading-relaxed">
                      <ReactMarkdown 
                        components={{
                          code({node, className, children, ...props}) {
                            return (
                              <code className={`${className || ''} text-blue-200`} {...props}>
                                {children}
                              </code>
                            )
                          }
                        }}
                      >
                          {`\`\`\`python${pattern.codeExample}\`\`\``}
                      </ReactMarkdown>
                   </div>
                </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
};

export default PatternDetailModal;
