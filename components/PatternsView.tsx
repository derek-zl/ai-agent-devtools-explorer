
import React, { useState } from 'react';
import { Pattern } from '../types';
import PatternDetailModal from './PatternDetailModal';
import { Workflow, Brain, Database, Users, ClipboardList, CheckCircle, Wrench, MessageSquareText, Lightbulb, Network, Code2, Container, FileJson, ListTodo, BrainCircuit, ScanEye, HardDrive, Share2, Ungroup, GitGraph, Table2, SearchCheck, Zap, BookOpen } from 'lucide-react';

interface PatternsViewProps {
  patterns: Pattern[];
  ui: any;
}

const PatternsView: React.FC<PatternsViewProps> = ({ patterns, ui }) => {
  const [selectedPattern, setSelectedPattern] = useState<Pattern | null>(null);
  const [showComparison, setShowComparison] = useState(false);

  const getIcon = (id: string) => {
    switch (id) {
      case 'explicit-function': return <Code2 className="text-pink-400" size={32} />;
      case 'structured-text': return <FileJson className="text-blue-400" size={32} />;
      case 'shadow-workspace': return <Container className="text-green-400" size={32} />;
      case 'ast-semantic': return <Brain className="text-yellow-400" size={32} />;
      case 'native-crdt': return <Network className="text-purple-400" size={32} />;
      case 'plan-execute': return <ListTodo className="text-cyan-400" size={32} />;
      case 'cot': return <BrainCircuit className="text-teal-400" size={32} />;
      case 'self-reflection': return <ScanEye className="text-amber-400" size={32} />;
      case 'memory-augmented': return <HardDrive className="text-indigo-400" size={32} />;
      case 'hierarchical': return <Share2 className="text-rose-400" size={32} />;
      case 'swarm': return <Ungroup className="text-lime-400" size={32} />;
      case 'state-machine': return <GitGraph className="text-orange-400" size={32} />;
      default: return <Workflow className="text-slate-400" size={32} />;
    }
  };

  return (
    <div className="animate-in fade-in duration-500 pb-10">
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-white mb-2">{ui.navPatterns}</h2>
        <p className="text-slate-400">{ui.headerSubtitle}</p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 mb-16">
        {patterns.map((pattern) => (
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
                  pattern.complexity === 'Very High' ? 'bg-purple-500/10 text-purple-400 border-purple-500/20' :
                  pattern.complexity === 'High' ? 'bg-red-500/10 text-red-400 border-red-500/20' :
                  pattern.complexity === 'Medium' ? 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20' :
                  'bg-green-500/10 text-green-400 border-green-500/20'
                }`}>
                {pattern.complexity} {ui.complexity}
              </span>
            </div>
            
            <h3 className="text-xl font-bold text-slate-100 mb-2 group-hover:text-indigo-300 transition-colors">{pattern.name}</h3>
            <p className="text-slate-400 mb-5 text-sm leading-relaxed line-clamp-3">{pattern.description}</p>
            
            <div className="mt-auto">
              <div className="bg-slate-900/50 rounded-lg p-4 mb-4 border border-slate-800/50">
                <p className="text-xs text-indigo-400 font-bold mb-1 uppercase tracking-wider flex items-center gap-1">
                   {ui.primaryUseCase}
                </p>
                <p className="text-sm text-slate-300 line-clamp-2">{pattern.useCase}</p>
              </div>

              <div className="flex flex-wrap gap-2">
                {pattern.tags.slice(0,3).map(tag => (
                  <span key={tag} className="px-2.5 py-1 text-xs rounded-md bg-slate-700/50 text-slate-400 border border-slate-700">
                    #{tag}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Comparison Section */}
      <div className="border-t border-slate-800 pt-10">
         <button 
            onClick={() => setShowComparison(!showComparison)}
            className="w-full flex items-center justify-between bg-slate-900/50 p-6 rounded-2xl border border-slate-800 hover:border-indigo-500/30 transition-all"
         >
            <div className="flex items-center gap-4">
                <div className="p-3 bg-indigo-500/10 rounded-xl text-indigo-400">
                   <Table2 size={24} />
                </div>
                <div className="text-left">
                    <h3 className="text-xl font-bold text-white">{ui.comparisonGuide}</h3>
                    <p className="text-slate-400 text-sm">Which architecture should you choose?</p>
                </div>
            </div>
            <div className={`text-indigo-400 transition-transform duration-300 ${showComparison ? 'rotate-180' : ''}`}>
                ▼
            </div>
         </button>

         {showComparison && (
             <div className="mt-6 animate-in slide-in-from-top-4 duration-300">
                 <div className="overflow-x-auto rounded-xl border border-slate-800 shadow-2xl">
                     <table className="w-full text-sm text-left text-slate-400">
                         <thead className="text-xs text-slate-200 uppercase bg-slate-900">
                             <tr>
                                 <th className="px-6 py-4 font-bold">Pattern</th>
                                 <th className="px-6 py-4 font-bold">Best Use Case</th>
                                 <th className="px-6 py-4 font-bold">Complexity</th>
                                 <th className="px-6 py-4 font-bold">Key Advantage</th>
                                 <th className="px-6 py-4 font-bold text-slate-500">Limitation</th>
                             </tr>
                         </thead>
                         <tbody className="divide-y divide-slate-800 bg-slate-900/50">
                             <tr className="hover:bg-slate-800/50">
                                 <td className="px-6 py-4 font-medium text-white">ReAct</td>
                                 <td className="px-6 py-4">Reasoning intensive</td>
                                 <td className="px-6 py-4 text-yellow-400">Medium</td>
                                 <td className="px-6 py-4">Transparent logic</td>
                                 <td className="px-6 py-4">Can loop/stuck</td>
                             </tr>
                             <tr className="hover:bg-slate-800/50">
                                 <td className="px-6 py-4 font-medium text-white">Plan-Execute</td>
                                 <td className="px-6 py-4">Long-term complex tasks</td>
                                 <td className="px-6 py-4 text-red-400">High</td>
                                 <td className="px-6 py-4">Clear goals/steps</td>
                                 <td className="px-6 py-4">Rigid planning</td>
                             </tr>
                             <tr className="hover:bg-slate-800/50">
                                 <td className="px-6 py-4 font-medium text-white">Multi-Agent</td>
                                 <td className="px-6 py-4">Specialized collaboration</td>
                                 <td className="px-6 py-4 text-purple-400">Very High</td>
                                 <td className="px-6 py-4">Efficiency via specialization</td>
                                 <td className="px-6 py-4">Complex coordination</td>
                             </tr>
                             <tr className="hover:bg-slate-800/50">
                                 <td className="px-6 py-4 font-medium text-white">RAG / Tool</td>
                                 <td className="px-6 py-4">Knowledge/External Data</td>
                                 <td className="px-6 py-4 text-yellow-400">Medium</td>
                                 <td className="px-6 py-4">Infinite knowledge</td>
                                 <td className="px-6 py-4">Dep. on retrieval quality</td>
                             </tr>
                             <tr className="hover:bg-slate-800/50">
                                 <td className="px-6 py-4 font-medium text-white">Chain-of-Thought</td>
                                 <td className="px-6 py-4">Math / Logic Puzzles</td>
                                 <td className="px-6 py-4 text-green-400">Low</td>
                                 <td className="px-6 py-4">Higher accuracy</td>
                                 <td className="px-6 py-4">More tokens/latency</td>
                             </tr>
                             <tr className="hover:bg-slate-800/50">
                                 <td className="px-6 py-4 font-medium text-white">Memory-Augmented</td>
                                 <td className="px-6 py-4">Long-term interaction</td>
                                 <td className="px-6 py-4 text-red-400">High</td>
                                 <td className="px-6 py-4">Personalization</td>
                                 <td className="px-6 py-4">Storage costs / noise</td>
                             </tr>
                             <tr className="hover:bg-slate-800/50">
                                 <td className="px-6 py-4 font-medium text-white">Swarm</td>
                                 <td className="px-6 py-4">Distributed optimization</td>
                                 <td className="px-6 py-4 text-red-400">High</td>
                                 <td className="px-6 py-4">Robustness</td>
                                 <td className="px-6 py-4">Hard to control</td>
                             </tr>
                         </tbody>
                     </table>
                 </div>

                 {/* Practical Advice */}
                 <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
                     <div className="bg-slate-900/50 border border-slate-800 p-6 rounded-xl">
                        <h4 className="flex items-center gap-2 font-bold text-white mb-4">
                           <SearchCheck className="text-green-400" />
                           When to use what?
                        </h4>
                        <ul className="space-y-3 text-sm text-slate-400">
                           <li className="flex items-start gap-2">
                              <span className="bg-slate-800 text-white px-2 py-0.5 rounded text-xs mt-0.5">Simple</span>
                              Use <b>Chain-of-Thought</b> or <b>ReAct</b>.
                           </li>
                           <li className="flex items-start gap-2">
                              <span className="bg-slate-800 text-white px-2 py-0.5 rounded text-xs mt-0.5">Data Heavy</span>
                              Use <b>RAG</b> or <b>Tool-Augmented</b>.
                           </li>
                           <li className="flex items-start gap-2">
                              <span className="bg-slate-800 text-white px-2 py-0.5 rounded text-xs mt-0.5">Long Term</span>
                              Use <b>Memory-Augmented</b>.
                           </li>
                           <li className="flex items-start gap-2">
                              <span className="bg-slate-800 text-white px-2 py-0.5 rounded text-xs mt-0.5">Complex</span>
                              Use <b>Plan-Execute</b> or <b>Hierarchical</b>.
                           </li>
                        </ul>
                     </div>

                     <div className="bg-indigo-900/20 border border-indigo-500/20 p-6 rounded-xl">
                        <h4 className="flex items-center gap-2 font-bold text-white mb-4">
                           <Zap className="text-yellow-400" />
                           Pro Tip: Mix & Match
                        </h4>
                        <p className="text-sm text-slate-300 leading-relaxed">
                           These patterns are not mutually exclusive. The most powerful agents often combine them.
                        </p>
                        <div className="mt-4 bg-slate-900 p-3 rounded-lg text-xs font-mono text-indigo-300 border border-indigo-500/30">
                           Hierarchy( Planner + RAG_Worker + CoT_Worker )
                        </div>
                        <p className="text-sm text-slate-400 mt-4">
                           For example, a Hierarchical Agent might use a Planner to delegate to a Worker that uses RAG to answer questions.
                        </p>
                     </div>
                 </div>
             </div>
         )}
      </div>

      <PatternDetailModal 
        pattern={selectedPattern} 
        onClose={() => setSelectedPattern(null)}
        ui={ui} 
      />
    </div>
  );
};

export default PatternsView;
