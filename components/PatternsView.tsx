import React, { useState } from 'react';
import { PATTERNS } from '../constants';
import { Pattern } from '../types';
import PatternDetailModal from './PatternDetailModal';
import { Workflow, Brain, Database, Users, ClipboardList, CheckCircle, Wrench, MessageSquareText, Lightbulb, Network, Code2, Container, FileJson } from 'lucide-react';

const PatternsView: React.FC = () => {
  const [selectedPattern, setSelectedPattern] = useState<Pattern | null>(null);

  const getIcon = (id: string) => {
    switch (id) {
      case 'explicit-function': return <Code2 className="text-pink-400" size={32} />;
      case 'structured-text': return <FileJson className="text-blue-400" size={32} />;
      case 'shadow-workspace': return <Container className="text-green-400" size={32} />;
      case 'ast-semantic': return <Brain className="text-yellow-400" size={32} />;
      case 'native-crdt': return <Network className="text-purple-400" size={32} />;
      default: return <Workflow className="text-slate-400" size={32} />;
    }
  };

  return (
    <div className="animate-in fade-in duration-500">
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-white mb-2">Advanced Agent Architectures</h2>
        <p className="text-slate-400">Deep dive into the modern design patterns powering production-grade AI agents.</p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {PATTERNS.map((pattern) => (
          <div 
            key={pattern.id} 
            onClick={() => setSelectedPattern(pattern)}
            className="bg-slate-800/50 border border-slate-700 rounded-2xl p-6 hover:border-indigo-500/30 transition-all hover:bg-slate-800/80 hover:shadow-xl hover:shadow-indigo-900/10 group cursor-pointer flex flex-col h-full"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="p-3 bg-slate-900 rounded-xl border border-slate-800 group-hover:scale-105 transition-transform duration-300">
                {getIcon(pattern.id)}
              </div>
              <span className={`px-3 py-1 rounded-full text-xs font-medium border ${
                  pattern.complexity === 'High' ? 'bg-red-500/10 text-red-400 border-red-500/20' :
                  pattern.complexity === 'Medium' ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20' :
                  'bg-green-500/10 text-green-400 border-green-500/20'
                }`}>
                {pattern.complexity} Complexity
              </span>
            </div>
            
            <h3 className="text-xl font-bold text-slate-100 mb-2 group-hover:text-indigo-300 transition-colors">{pattern.name}</h3>
            <p className="text-slate-400 mb-5 text-sm leading-relaxed">{pattern.description}</p>
            
            <div className="mt-auto">
              <div className="bg-slate-900/50 rounded-lg p-4 mb-4 border border-slate-800/50">
                <p className="text-xs text-indigo-400 font-bold mb-1 uppercase tracking-wider flex items-center gap-1">
                   Primary Use Case
                </p>
                <p className="text-sm text-slate-300">{pattern.useCase}</p>
              </div>

              <div className="flex flex-wrap gap-2">
                {pattern.tags.map(tag => (
                  <span key={tag} className="px-2.5 py-1 text-xs rounded-md bg-slate-700/50 text-slate-400 border border-slate-700">
                    #{tag}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>

      <PatternDetailModal 
        pattern={selectedPattern} 
        onClose={() => setSelectedPattern(null)} 
      />
    </div>
  );
};

export default PatternsView;