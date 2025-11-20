
import React from 'react';
import { CODING_TOOLS } from '../constants';
import { CodingToolType } from '../types';
import { Monitor, Terminal, Puzzle, Layers, ExternalLink, Zap, Search, GitBranch, Code2, Box } from 'lucide-react';

const IdeView: React.FC = () => {
  const getIcon = (type: CodingToolType) => {
    switch (type) {
      case 'IDE': return <Monitor className="text-blue-400" size={24} />;
      case 'CLI': return <Terminal className="text-green-400" size={24} />;
      case 'Extension': return <Puzzle className="text-purple-400" size={24} />;
      case 'Platform': return <Layers className="text-orange-400" size={24} />;
      default: return <Monitor size={24} />;
    }
  };

  const getMechanismIcon = (mech: string) => {
    if (mech.includes('Shadow')) return <Box size={16} />;
    if (mech.includes('AST') || mech.includes('Repo')) return <GitBranch size={16} />;
    if (mech.includes('CRDT')) return <Zap size={16} />;
    if (mech.includes('Context')) return <Search size={16} />;
    return <Code2 size={16} />;
  };

  return (
    <div className="animate-in fade-in duration-500">
      <div className="mb-10 text-center">
        <h2 className="text-3xl font-bold text-white mb-3">AI-Native Development Environments</h2>
        <p className="text-slate-400 max-w-3xl mx-auto">
          Modern coding tools go beyond simple autocomplete. They implement complex agent architectures like 
          <span className="text-indigo-400"> Shadow Workspaces</span>, 
          <span className="text-indigo-400"> AST Analysis</span>, and 
          <span className="text-indigo-400"> CRDT Collaboration</span> to write and fix code autonomously.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {CODING_TOOLS.map((tool) => (
          <div 
            key={tool.id}
            className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 hover:border-indigo-500/30 transition-all hover:shadow-lg hover:shadow-indigo-500/10 group flex flex-col h-full"
          >
            {/* Header */}
            <div className="flex justify-between items-start mb-4">
              <div className="flex items-center gap-3">
                <div className={`p-2 rounded-lg border border-slate-700 bg-slate-800`}>
                  {getIcon(tool.type)}
                </div>
                <div>
                  <h3 className="font-bold text-slate-100 text-lg group-hover:text-indigo-300 transition-colors">
                    {tool.name}
                  </h3>
                  <span className={`text-xs font-medium px-2 py-0.5 rounded ${
                    tool.type === 'IDE' ? 'bg-blue-900/30 text-blue-300' :
                    tool.type === 'CLI' ? 'bg-green-900/30 text-green-300' :
                    'bg-purple-900/30 text-purple-300'
                  }`}>
                    {tool.type}
                  </span>
                </div>
              </div>
              <a 
                href={tool.website} 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-slate-500 hover:text-white transition-colors"
              >
                <ExternalLink size={18} />
              </a>
            </div>

            {/* Core Mechanism Badge */}
            <div className="mb-4">
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md bg-indigo-950/30 border border-indigo-500/20 text-indigo-300 text-sm font-medium">
                {getMechanismIcon(tool.coreMechanism)}
                {tool.coreMechanism}
              </div>
            </div>

            {/* Description */}
            <p className="text-slate-400 text-sm mb-6 flex-grow leading-relaxed">
              {tool.description}
            </p>

            {/* Features */}
            <div className="mt-auto pt-4 border-t border-slate-800/50">
              <h4 className="text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Key Features</h4>
              <div className="flex flex-wrap gap-2">
                {tool.features.map(feature => (
                  <span key={feature} className="text-xs bg-slate-800 text-slate-300 px-2 py-1 rounded hover:bg-slate-700 transition-colors cursor-default">
                    {feature}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default IdeView;
