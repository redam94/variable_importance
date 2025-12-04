import { useState, useCallback, useEffect } from 'react'
import { useChatStore, selectCurrentWorkflowId } from '../stores/chatStore'
import { chatApi } from '../lib/api'
import type { RAGChunk, RAGStats } from '../types/api'
import {
  Search,
  Loader2,
  FileText,
  Database,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  FolderOpen,
} from 'lucide-react'
import clsx from 'clsx'

export function RAGSearchPage() {
  const workflowId = useChatStore(selectCurrentWorkflowId)
  const initSession = useChatStore((state) => state.initSession)
  
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<RAGChunk[]>([])
  const [stats, setStats] = useState<RAGStats | null>(null)
  const [loading, setLoading] = useState(false)
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null)
  
  // Init session on mount
  useEffect(() => {
    initSession('workflow-chat')
  }, [initSession])
  
  const fetchStats = useCallback(async () => {
    if (!workflowId) return
    try {
      const data = await chatApi.getRAGStats(workflowId)
      setStats(data)
    } catch {
      setStats(null)
    }
  }, [workflowId])
  
  useEffect(() => {
    fetchStats()
  }, [fetchStats])
  
  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim() || !workflowId) return
    
    setLoading(true)
    try {
      const response = await chatApi.queryRAG({
        query: query.trim(),
        workflow_id: workflowId,
        n_results: 10,
      })
      setResults(response.results)
      setExpandedIndex(response.results.length > 0 ? 0 : null)
    } catch {
      setResults([])
    } finally {
      setLoading(false)
    }
  }
  
  const getRelevanceColor = (score: number) => {
    if (score >= 0.8) return 'text-green-600 bg-green-100'
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-100'
    return 'text-gray-600 bg-gray-100'
  }
  
  return (
    <div className="p-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">🔍 RAG Search</h1>
        <div className="flex items-center gap-2 mt-1 text-gray-600">
          <FolderOpen size={16} />
          <span>{workflowId || 'No workflow selected'}</span>
        </div>
      </div>
      
      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-3 gap-4 mb-8">
          <div className="card p-4 flex items-center gap-4">
            <div className="w-10 h-10 rounded-lg bg-primary-100 text-primary-600 flex items-center justify-center">
              <FileText size={20} />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900">
                {stats.total_documents}
              </div>
              <div className="text-sm text-gray-500">Documents</div>
            </div>
          </div>
          
          <div className="card p-4 flex items-center gap-4">
            <div className="w-10 h-10 rounded-lg bg-primary-100 text-primary-600 flex items-center justify-center">
              <Database size={20} />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900">
                {stats.total_chunks}
              </div>
              <div className="text-sm text-gray-500">Chunks</div>
            </div>
          </div>
          
          <div className="card p-4">
            <div className="text-sm text-gray-500 mb-2">Document Types</div>
            <div className="flex flex-wrap gap-1">
              {Object.entries(stats.doc_types || {}).map(([type, count]) => (
                <span
                  key={type}
                  className="text-xs px-2 py-1 bg-gray-100 rounded-full"
                >
                  {type}: {count as number}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}
      
      {/* Search form */}
      <form onSubmit={handleSearch} className="mb-8">
        <div className="flex gap-3">
          <div className="flex-1 relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" size={20} />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search your knowledge base..."
              className="input-field pl-12 py-3 text-lg"
            />
          </div>
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="btn-primary px-8 flex items-center gap-2"
          >
            {loading ? <Loader2 size={20} className="animate-spin" /> : <Search size={20} />}
            Search
          </button>
        </div>
      </form>
      
      {/* Results */}
      {results.length > 0 && (
        <div className="space-y-4">
          <h3 className="text-lg font-medium text-gray-900">
            Found {results.length} results
          </h3>
          
          {results.map((result, idx) => (
            <div
              key={idx}
              className={clsx(
                'card overflow-hidden transition-all',
                expandedIndex === idx ? 'ring-2 ring-primary-200' : ''
              )}
            >
              <button
                onClick={() => setExpandedIndex(expandedIndex === idx ? null : idx)}
                className="w-full p-4 text-left flex items-start gap-4"
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    <span className={clsx(
                      'px-2 py-0.5 rounded text-xs font-medium',
                      getRelevanceColor(result.relevance_score)
                    )}>
                      {(result.relevance_score * 100).toFixed(0)}% match
                    </span>
                    <span className="text-xs text-gray-500">
                      {result.metadata?.doc_type || 'unknown'}
                    </span>
                  </div>
                  
                  <p className={clsx(
                    'text-gray-700',
                    expandedIndex !== idx && 'line-clamp-2'
                  )}>
                    {result.content}
                  </p>
                </div>
                
                <div className="text-gray-400">
                  {expandedIndex === idx ? (
                    <ChevronUp size={20} />
                  ) : (
                    <ChevronDown size={20} />
                  )}
                </div>
              </button>
              
              {expandedIndex === idx && result.metadata && (
                <div className="px-4 pb-4 pt-2 border-t border-gray-100">
                  <div className="text-xs text-gray-500 space-y-1">
                    {result.metadata.stage_name && (
                      <p>Stage: {result.metadata.stage_name}</p>
                    )}
                    {result.metadata.timestamp && (
                      <p>Created: {new Date(result.metadata.timestamp).toLocaleString()}</p>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
      
      {/* No results */}
      {!loading && results.length === 0 && query && (
        <div className="card p-12 text-center">
          <Search className="w-12 h-12 mx-auto text-gray-300 mb-4" />
          <p className="text-gray-600">No results found for "{query}"</p>
          <p className="text-sm text-gray-500 mt-2">
            Try different keywords or upload more documents
          </p>
        </div>
      )}
      
      {/* Initial state */}
      {!loading && results.length === 0 && !query && (
        <div className="card p-12 text-center">
          <Database className="w-12 h-12 mx-auto text-gray-300 mb-4" />
          <p className="text-gray-600">Enter a query to search your knowledge base</p>
          <p className="text-sm text-gray-500 mt-2">
            Results will be ranked by semantic relevance
          </p>
        </div>
      )}
      
      {/* Workflow context */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg text-sm text-gray-500 flex items-center justify-between">
        <span>
          Searching workflow: <code className="px-1 bg-gray-200 rounded">{workflowId || 'None'}</code>
        </span>
        <button
          onClick={fetchStats}
          className="flex items-center gap-1 text-primary-600 hover:underline"
        >
          <RefreshCw size={14} />
          Refresh stats
        </button>
      </div>
    </div>
  )
}