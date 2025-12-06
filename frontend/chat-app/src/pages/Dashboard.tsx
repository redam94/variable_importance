import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { useChatStore, selectCurrentWorkflowId, selectMessages } from '../stores/chatStore'
import { healthApi, chatApi } from '../lib/api'
import type { HealthResponse, RAGStats } from '../types/api'
import {
  MessageSquare,
  FileText,
  FolderOpen,
  Bot,
  Activity,
  Database,
  Cpu,
  RefreshCw,
} from 'lucide-react'

export function DashboardPage() {
  // Use selectors
  const workflowId = useChatStore(selectCurrentWorkflowId)
  const messages = useChatStore(selectMessages('workflow-chat'))
  const initSession = useChatStore((state) => state.initSession)
  
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [ragStats, setRagStats] = useState<RAGStats | null>(null)
  const [loading, setLoading] = useState(true)
  
  // Init session on mount
  useEffect(() => {
    initSession('workflow-chat')
  }, [initSession])
  
  const fetchData = async () => {
    setLoading(true)
    try {
      const [healthData, statsData] = await Promise.all([
        healthApi.check().catch(() => null),
        workflowId ? chatApi.getRAGStats(workflowId).catch(() => null) : null,
      ])
      setHealth(healthData)
      setRagStats(statsData)
    } finally {
      setLoading(false)
    }
  }
  
  useEffect(() => {
    fetchData()
  }, [workflowId])
  
  return (
    <div className="p-8 h-full min-h-screen overflow-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">🤖 Data Science Agent</h1>
        <div className="flex items-center gap-2 mt-2 text-gray-600">
          <FolderOpen size={18} />
          <span>Current workflow: <strong>{workflowId || 'None'}</strong></span>
        </div>
      </div>
      
      {/* Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="metric-card">
          <div className="text-4xl font-bold mb-2">{messages?.length ?? 0}</div>
          <div className="text-sm opacity-90">Chat Messages</div>
        </div>
        
        <div className="metric-card">
          <div className="text-4xl font-bold mb-2">{ragStats?.total_documents || 0}</div>
          <div className="text-sm opacity-90">RAG Documents</div>
        </div>
        
        <div className="metric-card">
          <div className="text-4xl font-bold mb-2">{ragStats?.total_chunks || 0}</div>
          <div className="text-sm opacity-90">Knowledge Chunks</div>
        </div>
        
        <div className="metric-card">
          <div className="flex items-center justify-center gap-2 text-2xl font-bold mb-2">
            <Activity size={24} />
            {health?.status === 'healthy' ? 'Online' : 'Degraded'}
          </div>
          <div className="text-sm opacity-90">System Status</div>
        </div>
      </div>
      
      {/* Quick Actions */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">🚀 Quick Actions</h2>
          <button
            onClick={fetchData}
            disabled={loading}
            className="btn-ghost flex items-center gap-2 text-gray-600"
          >
            <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Link to="/chat" className="card p-6 hover:shadow-lg transition-shadow group">
            <div className="w-12 h-12 gradient-bg rounded-xl flex items-center justify-center text-white mb-4 group-hover:scale-110 transition-transform">
              <MessageSquare size={24} />
            </div>
            <h3 className="font-semibold text-gray-900">Workflow Chat</h3>
            <p className="text-sm text-gray-500 mt-1">
              Run data analysis workflows
            </p>
          </Link>
          
          <Link to="/agent" className="card p-6 hover:shadow-lg transition-shadow group">
            <div className="w-12 h-12 gradient-bg rounded-xl flex items-center justify-center text-white mb-4 group-hover:scale-110 transition-transform">
              <Bot size={24} />
            </div>
            <h3 className="font-semibold text-gray-900">RAG Agent</h3>
            <p className="text-sm text-gray-500 mt-1">
              Chat with knowledge retrieval
            </p>
          </Link>
          
          <Link to="/documents" className="card p-6 hover:shadow-lg transition-shadow group">
            <div className="w-12 h-12 gradient-bg rounded-xl flex items-center justify-center text-white mb-4 group-hover:scale-110 transition-transform">
              <FileText size={24} />
            </div>
            <h3 className="font-semibold text-gray-900">Documents</h3>
            <p className="text-sm text-gray-500 mt-1">
              Upload and manage documents
            </p>
          </Link>
          
          <Link to="/workflows" className="card p-6 hover:shadow-lg transition-shadow group">
            <div className="w-12 h-12 gradient-bg rounded-xl flex items-center justify-center text-white mb-4 group-hover:scale-110 transition-transform">
              <FolderOpen size={24} />
            </div>
            <h3 className="font-semibold text-gray-900">Workflows</h3>
            <p className="text-sm text-gray-500 mt-1">
              Browse past analyses
            </p>
          </Link>
        </div>
      </div>
      
      {/* System Status */}
      <div className="card p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">⚙️ System Status</h2>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Cpu size={16} className="text-gray-400" />
              <span className="text-sm font-medium text-gray-600">Ollama</span>
            </div>
            <span className={`text-sm ${health?.ollama_connected ? 'text-green-600' : 'text-red-600'}`}>
              {health?.ollama_connected ? '● Online' : '○ Offline'}
            </span>
          </div>
          
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Database size={16} className="text-gray-400" />
              <span className="text-sm font-medium text-gray-600">RAG DB</span>
            </div>
            <span className={`text-sm ${ragStats ? 'text-green-600' : 'text-yellow-600'}`}>
              {ragStats ? '● Ready' : '○ Empty'}
            </span>
          </div>
          
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Activity size={16} className="text-gray-400" />
              <span className="text-sm font-medium text-gray-600">Default Model</span>
            </div>
            <span className="text-sm text-gray-600">
              {health?.default_model || "qwen3"}
            </span>
          </div>
          
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <FolderOpen size={16} className="text-gray-400" />
              <span className="text-sm font-medium text-gray-600">Workflow</span>
            </div>
            <span className="text-sm text-gray-600 truncate block">
              {workflowId || 'None'}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}