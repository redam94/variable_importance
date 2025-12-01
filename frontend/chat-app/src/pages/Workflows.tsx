import { useState, useEffect } from 'react'
import { useChatStore } from '../stores/chatStore'
import { workflowApi } from '../lib/api'
import type { Workflow } from '../types/api'
import {
  FolderOpen,
  RefreshCw,
  Check,
  Layers,
  FileOutput,
  Search,
} from 'lucide-react'
import clsx from 'clsx'

export function WorkflowsPage() {
  const { workflowId, setWorkflowId, clearMessages } = useChatStore()
  const [workflows, setWorkflows] = useState<Workflow[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  
  const fetchWorkflows = async () => {
    setLoading(true)
    try {
      const data = await workflowApi.listWorkflows()
      setWorkflows(data)
    } catch {
      setWorkflows([])
    } finally {
      setLoading(false)
    }
  }
  
  useEffect(() => {
    fetchWorkflows()
  }, [])
  
  const filteredWorkflows = workflows.filter((w) =>
    w.workflow_id.toLowerCase().includes(search.toLowerCase())
  )
  
  const handleLoadWorkflow = (id: string) => {
    console.log('Loading workflow:', id)
    setWorkflowId(id)
    clearMessages()
  }
  
  const totalStages = workflows.reduce((sum, w) => sum + w.stage_count, 0)
  const totalOutputs = workflows.reduce((sum, w) => sum + w.stage_count, 0)
  
  return (
    <div className="p-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">📚 Workflows</h1>
        <p className="mt-1 text-gray-600">
          Browse and switch between your analysis workflows
        </p>
      </div>
      
      {/* Summary */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        <div className="metric-card">
          <div className="text-3xl font-bold mb-1">{workflows.length}</div>
          <div className="text-sm opacity-90">Total Workflows</div>
        </div>
        <div className="metric-card">
          <div className="text-3xl font-bold mb-1">{totalStages}</div>
          <div className="text-sm opacity-90">Total Stages</div>
        </div>
        <div className="metric-card">
          <div className="text-3xl font-bold mb-1">{totalOutputs}</div>
          <div className="text-sm opacity-90">Total Outputs</div>
        </div>
      </div>
      
      {/* Search and actions */}
      <div className="flex gap-4 mb-6">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={18} />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search workflows..."
            className="input-field pl-10"
          />
        </div>
        <button
          onClick={fetchWorkflows}
          disabled={loading}
          className="btn-secondary flex items-center gap-2"
        >
          <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
          Refresh
        </button>
      </div>
      
      {/* Workflows list */}
      <div className="card divide-y divide-gray-100">
        {loading ? (
          <div className="p-12 text-center">
            <RefreshCw className="w-8 h-8 mx-auto text-primary-500 animate-spin mb-4" />
            <p className="text-gray-600">Loading workflows...</p>
          </div>
        ) : filteredWorkflows.length === 0 ? (
          <div className="p-12 text-center">
            <FolderOpen className="w-12 h-12 mx-auto text-gray-300 mb-4" />
            <p className="text-gray-600">
              {search ? 'No workflows match your search' : 'No workflows found'}
            </p>
          </div>
        ) : (
          filteredWorkflows.map((workflow) => {
            const isCurrent = workflow.workflow_id === workflowId
            
            return (
              <div
                key={workflow.workflow_id}
                className={clsx(
                  'p-4 hover:bg-gray-50 transition-colors',
                  isCurrent && 'bg-primary-50'
                )}
              >
                <div className="flex items-center gap-4">
                  <div
                    className={clsx(
                      'w-10 h-10 rounded-lg flex items-center justify-center',
                      isCurrent
                        ? 'gradient-bg text-white'
                        : 'bg-gray-100 text-gray-500'
                    )}
                  >
                    <FolderOpen size={20} />
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="font-medium text-gray-900 truncate">
                        {workflow.workflow_id}
                      </p>
                      {isCurrent && (
                        <span className="flex items-center gap-1 text-xs text-primary-600 bg-primary-100 px-2 py-0.5 rounded-full">
                          <Check size={12} />
                          Current
                        </span>
                      )}
                    </div>
                    
                    <div className="flex items-center gap-4 mt-1 text-sm text-gray-500">
                      <span className="flex items-center gap-1">
                        <Layers size={14} />
                        {workflow.stage_count} stages
                      </span>
                      <span className="flex items-center gap-1">
                        <FileOutput size={14} />
                        {workflow.stage_count} outputs
                      </span>
                    </div>
                  </div>
                  
                  {!isCurrent && (
                    <button
                      onClick={() => handleLoadWorkflow(workflow.workflow_id)}
                      className="btn-primary text-sm"
                    >
                      Load
                    </button>
                  )}
                </div>
              </div>
            )
          })
        )}
      </div>
      
      {/* Current workflow */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg text-sm text-gray-500">
        Current workflow: <code className="px-1 bg-gray-200 rounded">{workflowId}</code>
      </div>
    </div>
  )
}