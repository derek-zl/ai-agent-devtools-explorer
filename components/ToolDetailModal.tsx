import React, { useEffect, useState } from 'react';
import { Tool } from '../types';
import { PATTERNS } from '../constants';
import { generateToolComparison } from '../services/geminiService';
import { X, Sparkles, Link as LinkIcon, Github, LayoutTemplate, Check } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

interface ToolDetailModalProps {
  tool: Tool | null;
  onClose: () => void;
}

const ToolDetailModal: React.FC<ToolDetailModalProps> = ({ tool, onClose }) => {
  const [details, setDetails] = useState<string>('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (tool) {
      setLoading(true);
      setDetails(''); // Clear previous
      generateToolComparison(tool.name)
        .then(text => setDetails(text))
        .catch(() => setDetails('Failed to load details.'))
        .finally(() => setLoading(false));
    }
  }, [tool]);

  if (!tool) return null;

  const getPatternName = (id: string) => {
    const pattern = PATTERNS.find(p => p.id === id);
    return pattern ? pattern.name : id;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div 
        className="bg-slate-900 border border-slate-700 w-full max-w-3xl max-h-[85vh] rounded-2xl shadow-2xl overflow-hidden flex flex-col animate-in zoom-in-95 duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="p-6 border-b border-slate-800 flex justify-between items-start bg-slate-900">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <h2 className="text-2xl font-bold text-white">{tool.name}</h2>
              {tool.githubStars && (
                 <span className="px-2 py-0.5 bg-slate-800 text-slate-400 text-xs rounded-full border border-slate-700 flex items-center gap-1">
                   <Github size={12} /> {tool.githubStars}
                 </span>
              )}
            </div>
            <p className="text-slate-400 text-sm">{tool.description}</p>
          </div>
          <button 
            onClick={onClose}
            className="p-2 hover:bg-slate-800 rounded-full text-slate-400 hover:text-white transition-colors"
          >
            <X size={24} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 bg-slate-950">
          
          {/* Supported Patterns Section */}
          <div className="mb-8 bg-indigo-900/10 border border-indigo-500/20 rounded-xl p-4">
            <h3 className="flex items-center gap-2 text-indigo-400 font-semibold mb-3">
              <LayoutTemplate size={18} />
              Supported Patterns & Architectures
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {tool.supportedPatterns.map(patternId => (
                <div key={patternId} className="flex items-center gap-2 text-slate-300 text-sm">
                  <Check size={14} className="text-indigo-400" />
                  {getPatternName(patternId)}
                </div>
              ))}
            </div>
          </div>

          {loading ? (
            <div className="flex flex-col items-center justify-center h-48 space-y-4">
              <Sparkles className="animate-pulse text-indigo-500" size={48} />
              <p className="text-slate-400 animate-pulse">Asking Gemini about {tool.name}...</p>
            </div>
          ) : (
            <div className="prose prose-invert prose-slate max-w-none">
              <ReactMarkdown>{details}</ReactMarkdown>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-800 bg-slate-900 flex justify-end">
            <a 
              href={tool.website} 
              target="_blank" 
              rel="noopener noreferrer"
              className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-2 rounded-lg font-medium transition-colors"
            >
              <LinkIcon size={16} />
              Visit Official Site
            </a>
        </div>
      </div>
    </div>
  );
};

export default ToolDetailModal;
