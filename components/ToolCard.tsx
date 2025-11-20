
import React from 'react';
import { Tool, Language, Pattern } from '../types';
import { ExternalLink, Star, Terminal, Box, Code, LayoutTemplate } from 'lucide-react';

interface ToolCardProps {
  tool: Tool;
  onSelect: (tool: Tool) => void;
  ui: any;
  patterns: Pattern[];
}

const ToolCard: React.FC<ToolCardProps> = ({ tool, onSelect, ui, patterns }) => {
  const getLangColor = (lang: Language) => {
    switch (lang) {
      case Language.Python: return 'bg-blue-500/10 text-blue-400 border-blue-500/20';
      case Language.NodeJS: return 'bg-green-500/10 text-green-400 border-green-500/20';
      case Language.NoCode: return 'bg-purple-500/10 text-purple-400 border-purple-500/20';
      default: return 'bg-gray-500/10 text-gray-400 border-gray-500/20';
    }
  };

  const getPatternName = (id: string) => {
    const pattern = patterns.find(p => p.id === id);
    // Truncate long translated names for the card
    return pattern ? pattern.name.split(/[\s(（]/)[0] : id; 
  };

  return (
    <div 
      onClick={() => onSelect(tool)}
      className="group relative bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-5 hover:border-indigo-500/50 transition-all duration-300 cursor-pointer hover:shadow-lg hover:shadow-indigo-500/10 flex flex-col h-full"
    >
      <div className="flex justify-between items-start mb-4">
        <div className="p-3 bg-slate-800 rounded-lg border border-slate-700 group-hover:border-indigo-500/30 group-hover:bg-indigo-500/10 transition-colors">
          {tool.languages.includes(Language.Python) && !tool.languages.includes(Language.NodeJS) ? <Terminal size={24} className="text-blue-400" /> :
           tool.languages.includes(Language.NodeJS) && !tool.languages.includes(Language.Python) ? <Code size={24} className="text-green-400" /> :
           <Box size={24} className="text-orange-400" />}
        </div>
        {tool.githubStars && (
          <div className="flex items-center space-x-1 text-xs font-medium text-slate-400 bg-slate-900/50 px-2 py-1 rounded-full">
            <Star size={12} className="fill-amber-400 text-amber-400" />
            <span>{tool.githubStars}</span>
          </div>
        )}
      </div>

      <h3 className="text-xl font-bold text-slate-100 mb-2 group-hover:text-indigo-300 transition-colors">
        {tool.name}
      </h3>
      
      <p className="text-slate-400 text-sm mb-4 line-clamp-3 flex-grow">
        {tool.description}
      </p>

      {/* Supported Patterns Section */}
      <div className="mb-4">
        <div className="flex items-center gap-1 text-xs text-slate-500 mb-2 font-medium uppercase tracking-wide">
          <LayoutTemplate size={12} />
          <span>{ui.supportedPatterns}</span>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {tool.supportedPatterns.slice(0, 3).map(patternId => (
             <span key={patternId} className="px-2 py-0.5 rounded bg-indigo-900/30 border border-indigo-500/20 text-indigo-300 text-[10px]">
                {getPatternName(patternId)}
             </span>
          ))}
          {tool.supportedPatterns.length > 3 && (
            <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-500 text-[10px]">+{tool.supportedPatterns.length - 3}</span>
          )}
        </div>
      </div>

      <div className="flex flex-wrap gap-2 mb-4 pt-4 border-t border-slate-700/50">
        {tool.languages.map((lang) => (
          <span key={lang} className={`px-2 py-1 rounded-md text-xs font-medium border ${getLangColor(lang)}`}>
            {lang}
          </span>
        ))}
      </div>

      <div className="flex flex-wrap gap-2">
        {tool.tags.slice(0, 3).map((tag) => (
          <span key={tag} className="px-2 py-1 rounded-md text-xs bg-slate-700/50 text-slate-300">
            #{tag}
          </span>
        ))}
      </div>

      <div className="absolute top-5 right-5 opacity-0 group-hover:opacity-100 transition-opacity">
        <ExternalLink size={16} className="text-slate-400" />
      </div>
    </div>
  );
};

export default ToolCard;
