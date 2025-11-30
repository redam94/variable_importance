import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { useChatStore } from '../stores/chatStore'
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
  const { workflowId, messages } = useChatStore()
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [ragStats, setRagStats] = useState<RAGStats | null>(null)
  const [loading, setLoading] = useState(true)
  
  const fetchData = async () => {
    setLoading(true)
    try {
      const [healthData, statsData] = await Promise.all([
        healthApi.check().catch(() => null),
        chatApi.getRAGStats(workflowId).catch(() => null),
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
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold gradient-text">🤖 Data Science Agent</h1>
        <p className="mt-2 text-gray-600">
          AI-powered analysis with intelligent context retrieval
        </p>
      </div>
      
      {/* Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <div className="metric-card">
          <div className="text-4xl font-bold mb-2">{messages.length}</div>
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
        
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Link
            to="/chat"
            className="card p-6 hover:border-primary-300 hover:shadow-md transition-all group"
          >
            <div className="w-12 h-12 rounded-xl gradient-bg flex items-center justify-center text-white mb-4 group-hover:scale-110 transition-transform">
              <MessageSquare size={24} />
            </div>
            <h3 className="font-semibold text-gray-900">Start Chat</h3>
            <p className="text-sm text-gray-500 mt-1">Run data analysis workflows</p>
          </Link>
          
          <Link
            to="/agent"
            className="card p-6 hover:border-primary-300 hover:shadow-md transition-all group"
          >
            <div className="w-12 h-12 rounded-xl gradient-bg flex items-center justify-center text-white mb-4 group-hover:scale-110 transition-transform">
              <Bot size={24} />
            </div>
            <h3 className="font-semibold text-gray-900">RAG Agent</h3>
            <p className="text-sm text-gray-500 mt-1">Chat with knowledge retrieval</p>
          </Link>
          
          <Link
            to="/documents"
            className="card p-6 hover:border-primary-300 hover:shadow-md transition-all group"
          >
            <div className="w-12 h-12 rounded-xl gradient-bg flex items-center justify-center text-white mb-4 group-hover:scale-110 transition-transform">
              <FileText size={24} />
            </div>
            <h3 className="font-semibold text-gray-900">Upload Documents</h3>
            <p className="text-sm text-gray-500 mt-1">Add PDFs, text, or URLs</p>
          </Link>
          
          <Link
            to="/workflows"
            className="card p-6 hover:border-primary-300 hover:shadow-md transition-all group"
          >
            <div className="w-12 h-12 rounded-xl gradient-bg flex items-center justify-center text-white mb-4 group-hover:scale-110 transition-transform">
              <FolderOpen size={24} />
            </div>
            <h3 className="font-semibold text-gray-900">Browse Workflows</h3>
            <p className="text-sm text-gray-500 mt-1">View past analyses</p>
          </Link>
        </div>
      </div>
      
      {/* System Status */}
      <div className="card p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">⚙️ System Status</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="flex items-center gap-4">
            <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${health?.ollama_reachable ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'}`}>
              <Cpu size={20} />
            </div>
            <div>
              <p className="font-medium text-gray-900">Ollama</p>
              <p className="text-sm text-gray-500">
                {health?.ollama_reachable ? 'Connected' : 'Disconnected'}
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${health?.rag_enabled ? 'bg-green-100 text-green-600' : 'bg-red-100 text-red-600'}`}>
              <Database size={20} />
            </div>
            <div>
              <p className="font-medium text-gray-900">RAG System</p>
              <p className="text-sm text-gray-500">
                {health?.rag_enabled ? 'Active' : 'Inactive'}
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-lg bg-primary-100 text-primary-600 flex items-center justify-center">
              <Activity size={20} />
            </div>
            <div>
              <p className="font-medium text-gray-900">API Version</p>
              <p className="text-sm text-gray-500">{health?.version || 'Unknown'}</p>
            </div>
          </div>
        </div>
        
        {/* Current Workflow */}
        <div className="mt-6 pt-6 border-t border-gray-100">
          <p className="text-sm text-gray-500">
            Current Workflow: <code className="px-2 py-1 bg-gray-100 rounded text-gray-700">{workflowId}</code>
          </p>
        </div>
      </div>
    </div>
  )
}