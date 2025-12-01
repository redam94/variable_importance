import { useRef, useEffect, useCallback, useState } from 'react'
import { useChatStore } from '../stores/chatStore'
import { useChatWebSocket } from '../hooks/useWebSocket'
import { workflowApi } from '../lib/api'
import type { WorkflowResponse, WSMessage } from '../types/api'
import { ChatMessage } from '../components/chat/ChatMessage'
import { ChatInput } from '../components/chat/ChatInput'
import { WorkflowProgress } from '../components/workflow/workflowProgress'
import type { WorkflowStage } from '../components/workflow/workflowProgress'
import { WorkflowOutput } from '../components/workflow/WorkflowOutput'
import { FileUploadZone } from '../components/workflow/FileUploadZone'
import {
  Trash2,
  Settings,
  Wifi,
  WifiOff,
  RefreshCw,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  Database,
} from 'lucide-react'
import clsx from 'clsx'

interface WorkflowResult {
  response: WorkflowResponse
  timestamp: Date
}

export function WorkflowChatPage() {
  const {
    workflowId,
    messages,
    isStreaming,
    error,
    addMessage,
    startStreaming,
    stopStreaming,
    clearMessages,
    setError,
    setWorkflowId,
  } = useChatStore()

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [showSettings, setShowSettings] = useState(false)
  const [copied, setCopied] = useState(false)

  // File upload state
  const [dataFile, setDataFile] = useState<File | null>(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadSuccess, setUploadSuccess] = useState(false)
  const [dataPath, setDataPath] = useState<string | null>(null)

  // Workflow state
  const [workflowStages, setWorkflowStages] = useState<WorkflowStage[]>([])
  const [workflowResults, setWorkflowResults] = useState<WorkflowResult[]>([])
  const [showUpload, setShowUpload] = useState(true)

  // WebSocket for progress tracking
  const { isConnected, messages: wsMessages, currentStage, progress } = useChatWebSocket(workflowId)

  // Update stages from WebSocket messages
  useEffect(() => {
    if (wsMessages.length === 0) return

    const latestMessage = wsMessages[wsMessages.length - 1]
    updateStagesFromWS(latestMessage)
  }, [wsMessages])

  const updateStagesFromWS = (msg: WSMessage) => {
    if (msg.type === 'stage_start' && msg.stage) {
      setWorkflowStages((prev) => {
        const existing = prev.find((s) => s.name === msg.stage)
        if (existing) {
          return prev.map((s) =>
            s.name === msg.stage ? { ...s, status: 'running', message: msg.message } : s
          )
        }
        return [...prev, { name: msg.stage, status: 'running', message: msg.message }]
      })
    } else if (msg.type === 'stage_end' && msg.stage) {
      setWorkflowStages((prev) =>
        prev.map((s) =>
          s.name === msg.stage ? { ...s, status: 'completed', message: msg.message } : s
        )
      )
    } else if (msg.type === 'error' && msg.stage) {
      setWorkflowStages((prev) =>
        prev.map((s) =>
          s.name === msg.stage ? { ...s, status: 'error', message: msg.message } : s
        )
      )
    }
  }

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, workflowResults])

  // Handle file upload
  const handleFileSelect = async (file: File | null) => {
    if (!file) {
      setDataFile(null)
      setDataPath(null)
      setUploadSuccess(false)
      return
    }

    setDataFile(file)
    setIsUploading(true)
    setUploadProgress(0)
    setUploadSuccess(false)
    setError(null)

    try {
      const response = await workflowApi.uploadData(file, (progress) => {
        setUploadProgress(progress)
      })

      setUploadSuccess(true)
      setDataPath(response.file_path)
      addMessage({
        role: 'assistant',
        content: `📁 Uploaded **${response.filename}** (${(response.file_size / 1024).toFixed(1)} KB) - Ready for analysis`,
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
      setDataFile(null)
    } finally {
      setIsUploading(false)
    }
  }

  // Handle workflow execution
  const handleSend = useCallback(
    async (message: string) => {
      addMessage({ role: 'user', content: message })
      startStreaming()
      setError(null)
      setWorkflowStages([])

      try {
        const response = await workflowApi.run({
          workflow_id: workflowId,
          query: message,
          data_path: dataPath || undefined,
          stage_name: 'analysis',
          web_search_enabled: false,
        })

        // Store result
        setWorkflowResults((prev) => [...prev, { response, timestamp: new Date() }])

        // Add summary as assistant message
        addMessage({
          role: 'assistant',
          content: response.summary || response.message.content,
        })

        // Mark all stages complete
        setWorkflowStages((prev) =>
          prev.map((s) => (s.status === 'running' ? { ...s, status: 'completed' } : s))
        )
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Workflow failed')
        setWorkflowStages((prev) =>
          prev.map((s) => (s.status === 'running' ? { ...s, status: 'error' } : s))
        )
      } finally {
        stopStreaming()
      }
    },
    [workflowId, dataPath, addMessage, startStreaming, stopStreaming, setError]
  )

  const handleStop = useCallback(() => {
    stopStreaming()
  }, [stopStreaming])

  const handleNewWorkflow = () => {
    const now = new Date()
    const id = `workflow_${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}`
    setWorkflowId(id)
    clearMessages()
    setDataFile(null)
    setDataPath(null)
    setUploadSuccess(false)
    setWorkflowResults([])
    setWorkflowStages([])
  }

  const handleCopyWorkflowId = async () => {
    await navigator.clipboard.writeText(workflowId)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Get latest workflow result for display
  const latestResult = workflowResults[workflowResults.length - 1]?.response

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="flex-shrink-0 flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-white">
        <div className="flex items-center gap-3">
          <div className="text-2xl">🔬</div>
          <div>
            <h1 className="text-lg font-semibold text-gray-900">Workflow Analysis</h1>
            <p className="text-xs text-gray-500 font-mono">{workflowId}</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {/* Connection status */}
          <div
            className={clsx(
              'flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium',
              isConnected ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'
            )}
          >
            {isConnected ? <Wifi size={14} /> : <WifiOff size={14} />}
            {isConnected ? 'Connected' : 'Disconnected'}
          </div>

          {/* Data file indicator */}
          {dataFile && (
            <div className="flex items-center gap-1.5 px-2.5 py-1 bg-blue-100 text-blue-700 rounded-full text-xs font-medium">
              <Database size={14} />
              {dataFile.name.slice(0, 15)}
              {dataFile.name.length > 15 && '...'}
            </div>
          )}

          <button onClick={() => setShowSettings(!showSettings)} className="btn-ghost" title="Settings">
            <Settings size={20} />
          </button>

          <button
            onClick={clearMessages}
            className="btn-ghost text-red-500 hover:bg-red-50"
            title="Clear messages"
          >
            <Trash2 size={20} />
          </button>
        </div>
      </header>

      {/* Settings panel */}
      {showSettings && (
        <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
          <div className="max-w-4xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Workflow ID</label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={workflowId}
                  onChange={(e) => setWorkflowId(e.target.value)}
                  className="input-field text-sm font-mono"
                />
                <button onClick={handleCopyWorkflowId} className="btn-secondary px-3" title="Copy">
                  {copied ? <Check size={16} /> : <Copy size={16} />}
                </button>
              </div>
            </div>
            <div className="flex items-end">
              <button onClick={handleNewWorkflow} className="btn-primary">
                🔄 New Workflow
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Error banner */}
      {error && (
        <div className="px-6 py-3 bg-red-50 border-b border-red-200 text-red-700 text-sm">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Main content */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* File Upload Section */}
          <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
            <button
              onClick={() => setShowUpload(!showUpload)}
              className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 hover:bg-gray-100 transition-colors"
            >
              <span className="flex items-center gap-2 text-sm font-medium text-gray-700">
                <Database size={16} />
                Data File
                {dataFile && (
                  <span className="px-2 py-0.5 bg-green-100 text-green-700 rounded-full text-xs">
                    {dataFile.name}
                  </span>
                )}
              </span>
              {showUpload ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>
            {showUpload && (
              <div className="p-4">
                <FileUploadZone
                  onFileSelect={handleFileSelect}
                  currentFile={dataFile}
                  uploadProgress={uploadProgress}
                  isUploading={isUploading}
                  uploadSuccess={uploadSuccess}
                  disabled={isStreaming}
                />
              </div>
            )}
          </div>

          {/* Workflow Progress */}
          {isStreaming && (
            <WorkflowProgress
              stages={workflowStages}
              currentStage={currentStage}
              progress={progress}
              isRunning={isStreaming}
            />
          )}

          {/* Messages */}
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center text-center py-12">
              <div className="w-20 h-20 gradient-bg rounded-2xl flex items-center justify-center text-white text-4xl mb-6">
                🔬
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-2">Run a Data Analysis</h2>
              <p className="text-gray-500 max-w-md mb-6">
                Upload a data file and ask questions. The workflow will analyze your data, generate
                code, execute it, and provide insights.
              </p>
              <div className="flex flex-wrap gap-2 justify-center max-w-xl">
                {[
                  'Analyze the trends in my data',
                  'Show correlations between columns',
                  'Create a summary statistics report',
                  'Find outliers in the dataset',
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => handleSend(suggestion)}
                    className="px-4 py-2 bg-gray-100 hover:bg-gray-200 rounded-full text-sm text-gray-700 transition-colors"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {messages.map((msg, idx) => (
                <ChatMessage key={idx} role={msg.role} content={msg.content} />
              ))}

              {/* Show latest workflow output */}
              {latestResult && (
                <WorkflowOutput
                  code={latestResult.code_executed}
                  plots={latestResult.plots || undefined}
                  summary={latestResult.summary}
                  error={latestResult.error}
                  actionTaken={latestResult.action_taken}
                />
              )}

              {isStreaming && (
                <div className="flex items-center gap-2 text-gray-500 text-sm">
                  <RefreshCw size={16} className="animate-spin" />
                  Running workflow...
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>
      </div>

      {/* Input */}
      <ChatInput
        onSend={handleSend}
        onStop={handleStop}
        isLoading={isStreaming}
        placeholder={dataFile ? `Ask about ${dataFile.name}...` : 'Upload a file or ask a question...'}
      />
    </div>
  )
}