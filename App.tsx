
import React, { useState, useMemo } from 'react';
import { TOOLS } from './constants';
import { Language, Tool } from './types';
import ToolCard from './components/ToolCard';
import ChatOverlay from './components/ChatOverlay';
import ToolDetailModal from './components/ToolDetailModal';
import PatternsView from './components/PatternsView';
import IdeView from './components/IdeView';
import BuildGuideView from './components/BuildGuideView';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ZAxis } from 'recharts';
import { Search, LayoutGrid, BarChart3, Cpu, Layers, Wrench, MonitorPlay, Hammer } from 'lucide-react';

const App: React.FC = () => {
  const [currentPage, setCurrentPage] = useState<'tools' | 'patterns' | 'ides' | 'build'>('tools');
  const [selectedLanguage, setSelectedLanguage] = useState<Language | 'All'>('All');
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'chart'>('grid');
  const [selectedTool, setSelectedTool] = useState<Tool | null>(null);

  const filteredTools = useMemo(() => {
    return TOOLS.filter(tool => {
      const matchesLang = selectedLanguage === 'All' || tool.languages.includes(selectedLanguage as Language);
      const matchesSearch = tool.name.toLowerCase().includes(searchQuery.toLowerCase()) || 
                            tool.tags.some(t => t.toLowerCase().includes(searchQuery.toLowerCase()));
      return matchesLang && matchesSearch;
    });
  }, [selectedLanguage, searchQuery]);

  // Data for Chart View
  const chartData = filteredTools.map(t => ({
    x: t.complexity,
    y: t.power,
    z: 100, // Bubble size
    name: t.name,
    tool: t // Pass full object for tooltip
  }));

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-slate-800 border border-slate-700 p-3 rounded-lg shadow-xl">
          <p className="font-bold text-white mb-1">{data.name}</p>
          <p className="text-xs text-slate-400">Power: {data.y}/10</p>
          <p className="text-xs text-slate-400">Complexity: {data.x}/10</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200 selection:bg-indigo-500/30 pb-20">
      {/* Decorative Background Elements */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden pointer-events-none z-0">
        <div className="absolute -top-[20%] -left-[10%] w-[50%] h-[50%] bg-indigo-900/20 rounded-full blur-[120px]" />
        <div className="absolute top-[40%] -right-[10%] w-[40%] h-[40%] bg-blue-900/10 rounded-full blur-[100px]" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        
        {/* Header Section */}
        <header className="mb-8 text-center">
          <div className="inline-flex items-center justify-center p-3 mb-4 bg-slate-800/50 border border-slate-700 rounded-2xl backdrop-blur-md">
            <Cpu className="text-indigo-400 mr-3" size={32} />
            <h1 className="text-3xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
              AI Agent Stack
            </h1>
          </div>
          <p className="text-lg text-slate-400 max-w-2xl mx-auto mb-8">
            Explore the best tools, frameworks, and architectural patterns for building autonomous agents.
          </p>

          {/* Main Navigation */}
          <nav className="inline-flex flex-wrap justify-center bg-slate-900/50 p-1.5 rounded-xl border border-slate-800 backdrop-blur-sm gap-1">
            <button
              onClick={() => setCurrentPage('tools')}
              className={`flex items-center gap-2 px-4 sm:px-6 py-2.5 rounded-lg font-medium transition-all duration-200 ${
                currentPage === 'tools' 
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-900/50' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              <Wrench size={18} />
              Tools Explorer
            </button>
            <button
              onClick={() => setCurrentPage('patterns')}
              className={`flex items-center gap-2 px-4 sm:px-6 py-2.5 rounded-lg font-medium transition-all duration-200 ${
                currentPage === 'patterns' 
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-900/50' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              <Layers size={18} />
              Agent Patterns
            </button>
            <button
              onClick={() => setCurrentPage('ides')}
              className={`flex items-center gap-2 px-4 sm:px-6 py-2.5 rounded-lg font-medium transition-all duration-200 ${
                currentPage === 'ides' 
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-900/50' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              <MonitorPlay size={18} />
              AI IDEs
            </button>
             <button
              onClick={() => setCurrentPage('build')}
              className={`flex items-center gap-2 px-4 sm:px-6 py-2.5 rounded-lg font-medium transition-all duration-200 ${
                currentPage === 'build' 
                  ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-900/50' 
                  : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
            >
              <Hammer size={18} />
              Build Guide
            </button>
          </nav>
        </header>

        {/* Conditional Content */}
        {currentPage === 'tools' ? (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
            {/* Controls Section */}
            <div className="flex flex-col md:flex-row justify-between items-center gap-6 mb-10 sticky top-4 z-30 p-4 bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl shadow-2xl">
              
              {/* Filter Tabs */}
              <div className="flex p-1 bg-slate-800 rounded-xl w-full md:w-auto overflow-x-auto">
                {(['All', Language.Python, Language.NodeJS, Language.NoCode] as const).map((lang) => (
                  <button
                    key={lang}
                    onClick={() => setSelectedLanguage(lang)}
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all whitespace-nowrap ${
                      selectedLanguage === lang 
                        ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-900/50' 
                        : 'text-slate-400 hover:text-white hover:bg-slate-700/50'
                    }`}
                  >
                    {lang}
                  </button>
                ))}
              </div>

              <div className="flex w-full md:w-auto gap-4">
                {/* Search */}
                <div className="relative flex-grow md:flex-grow-0 md:w-64">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
                  <input
                    type="text"
                    placeholder="Search tools..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 bg-slate-800 border border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/50 text-sm text-white placeholder-slate-500"
                  />
                </div>

                {/* View Toggle */}
                <div className="flex bg-slate-800 rounded-xl p-1">
                  <button
                    onClick={() => setViewMode('grid')}
                    className={`p-2 rounded-lg transition-all ${viewMode === 'grid' ? 'bg-slate-700 text-white' : 'text-slate-500 hover:text-slate-300'}`}
                  >
                    <LayoutGrid size={18} />
                  </button>
                  <button
                    onClick={() => setViewMode('chart')}
                    className={`p-2 rounded-lg transition-all ${viewMode === 'chart' ? 'bg-slate-700 text-white' : 'text-slate-500 hover:text-slate-300'}`}
                  >
                    <BarChart3 size={18} />
                  </button>
                </div>
              </div>
            </div>

            {/* Tools Content */}
            {viewMode === 'grid' ? (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filteredTools.map(tool => (
                  <ToolCard key={tool.id} tool={tool} onSelect={setSelectedTool} />
                ))}
              </div>
            ) : (
              <div className="h-[600px] bg-slate-900/50 border border-slate-800 rounded-2xl p-6 relative overflow-hidden">
                 <h3 className="text-center text-slate-400 mb-4 text-sm uppercase tracking-wider font-semibold">
                   Complexity vs Power Trade-off
                 </h3>
                 <ResponsiveContainer width="100%" height="100%">
                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis 
                        type="number" 
                        dataKey="x" 
                        name="Complexity" 
                        unit="/10" 
                        stroke="#64748b" 
                        label={{ value: 'Complexity (Learning Curve)', position: 'insideBottom', offset: -10, fill: '#64748b' }} 
                        domain={[0, 10]}
                      />
                      <YAxis 
                        type="number" 
                        dataKey="y" 
                        name="Power" 
                        unit="/10" 
                        stroke="#64748b" 
                        label={{ value: 'Power / Flexibility', angle: -90, position: 'insideLeft', fill: '#64748b' }} 
                        domain={[0, 10]}
                      />
                      <ZAxis type="number" dataKey="z" range={[100, 100]} />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} content={<CustomTooltip />} />
                      <Scatter data={chartData} onClick={(data) => setSelectedTool(data.tool)}>
                        {chartData.map((entry, index) => (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={entry.tool.languages.includes(Language.Python) && entry.tool.languages.includes(Language.NodeJS) ? '#818cf8' : entry.tool.languages.includes(Language.Python) ? '#3b82f6' : '#22c55e'} 
                            className="cursor-pointer hover:opacity-80 transition-opacity"
                          />
                        ))}
                      </Scatter>
                    </ScatterChart>
                 </ResponsiveContainer>
                 <div className="absolute bottom-4 right-4 text-xs text-slate-500 bg-slate-900/80 p-2 rounded border border-slate-800">
                    <span className="text-blue-500">● Python</span> <span className="text-green-500 ml-2">● Node.js</span> <span className="text-indigo-400 ml-2">● Both</span>
                 </div>
              </div>
            )}

            {filteredTools.length === 0 && (
              <div className="text-center py-20">
                <p className="text-slate-500 text-lg">No tools found matching your criteria.</p>
                <button 
                  onClick={() => {setSearchQuery(''); setSelectedLanguage('All');}}
                  className="mt-4 text-indigo-400 hover:text-indigo-300 underline"
                >
                  Clear filters
                </button>
              </div>
            )}
          </div>
        ) : currentPage === 'patterns' ? (
          <PatternsView />
        ) : currentPage === 'ides' ? (
          <IdeView />
        ) : (
          <BuildGuideView />
        )}

      </div>

      {/* Overlays */}
      <ChatOverlay />
      <ToolDetailModal tool={selectedTool} onClose={() => setSelectedTool(null)} />
    </div>
  );
};

export default App;
