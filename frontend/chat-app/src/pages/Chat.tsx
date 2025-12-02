import { useRef, useEffect, useState, useCallback } from 'react'
import { useWorkflowChat, type WorkflowOptions } from '../hooks/useWorkflowChat'
import { useChatStore } from '../stores/chatStore'
import { workflowApi } from '../lib/api'
import { ChatMessage } from '../components/chat/ChatMessage'
import { RAGAgentMessage } from '../components/chat/RAGAgentMessage'
import { ChatInput } from '../components/chat/ChatInput'
import { WorkflowProgress } from '../components/workflow/WorkflowProgress'
import { FileUploadZone } from '../components/workflow/FileUploadZone'
import {
  Trash2,
  Wifi,
  WifiOff,
  RefreshCw,
  Settings,
  Database,
  Search,
  BookOpen,
  X,
} from 'lucide-react'
import clsx from 'clsx'
import type { ChatMessage as ChatMessageType } from '../types/ws'

export function WorkflowChatPage() {
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [showSettings, setShowSettings] = useState(false)

  // Get workflowId from store (persisted)
  const { workflowId } = useChatStore()

  // File upload state
  const [dataFile, setDataFile] = useState<File | null>(null)
  const [dataPath, setDataPath] = useState<string | null>(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadSuccess, setUploadSuccess] = useState(false)

  // Workflow options
  const [ragEnabled, setRagEnabled] = useState(true)
  const [webSearchEnabled, setWebSearchEnabled] = useState(false)

  const {
    isConnected,
    isRunning,
    messages,
    currentStage,
    progress,
    startWorkflow,
    clearMessages,
    reconnect,
  } = useWorkflowChat({
    workflowId,
    onComplete: (taskId) => {
      console.log('Workflow complete:', taskId)
    },
    onError: (error) => {
      console.error('Workflow error:', error)
    },
  })

  // Auto-scroll on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = async (query: string) => {
    try {
      const options: WorkflowOptions = {
        dataPath,
        ragEnabled,
        webSearchEnabled,
      }
      await startWorkflow(query, options)
    } catch (error) {
      console.error('Failed to start workflow:', error)
    }
  }

  const handleFileSelect = useCallback(async (file: File | null) => {
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

    try {
      const response = await workflowApi.uploadData(file, (progress) => {
        setUploadProgress(progress)
      })
      setDataPath(response.file_path)
      setUploadSuccess(true)
    } catch (error) {
      console.error('Upload failed:', error)
      setDataFile(null)
      setUploadSuccess(false)
    } finally {
      setIsUploading(false)
    }
  }, [])

  const handleClearFile = useCallback(() => {
    setDataFile(null)
    setDataPath(null)
    setUploadSuccess(false)
  }, [])

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-white">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold text-gray-900">Workflow Chat</h1>

          {/* Connection status */}
          <div
            className={clsx(
              'flex items-center gap-1.5 px-2 py-1 rounded-full text-xs',
              isConnected ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
            )}
          >
            {isConnected ? <Wifi size={12} /> : <WifiOff size={12} />}
            {isConnected ? 'Connected' : 'Disconnected'}
          </div>

          {/* Workflow ID */}
          <span className="text-xs text-gray-400 font-mono">{workflowId}</span>
        </div>

        <div className="flex items-center gap-2">
          {/* Settings toggle */}
          <button
            onClick={() => setShowSettings(!showSettings)}
            className={clsx(
              'btn-ghost p-2',
              showSettings && 'bg-gray-100'
            )}
            title="Settings"
          >
            <Settings size={18} />
          </button>

          {!isConnected && (
            <button onClick={reconnect} className="btn-secondary flex items-center gap-2">
              <RefreshCw size={16} />
              Reconnect
            </button>
          )}

          <button
            onClick={() => {
              clearMessages()
              handleClearFile()
            }}
            className="btn-ghost text-gray-600 hover:text-red-600"
            title="Clear chat"
          >
            <Trash2 size={18} />
          </button>
        </div>
      </header>

      {/* Settings panel */}
      {showSettings && (
        <div className="px-6 py-4 border-b border-gray-200 bg-gray-50">
          <div className="flex flex-wrap gap-6">
            {/* RAG Toggle */}
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={ragEnabled}
                onChange={(e) => setRagEnabled(e.target.checked)}
                className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
              />
              <BookOpen size={16} className="text-gray-500" />
              <span className="text-sm text-gray-700">RAG Context</span>
            </label>

            {/* Web Search Toggle */}
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={webSearchEnabled}
                onChange={(e) => setWebSearchEnabled(e.target.checked)}
                className="w-4 h-4 text-primary-600 rounded focus:ring-primary-500"
              />
              <Search size={16} className="text-gray-500" />
              <span className="text-sm text-gray-700">Web Search</span>
            </label>

            {/* Data file indicator */}
            {dataPath && (
              <div className="flex items-center gap-2 text-sm text-green-600">
                <Database size={16} />
                <span className="truncate max-w-xs">{dataFile?.name}</span>
                <button
                  onClick={handleClearFile}
                  className="p-0.5 hover:bg-gray-200 rounded"
                >
                  <X size={14} />
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Progress bar */}
      {isRunning && <WorkflowProgress stage={currentStage || "analysis"} progress={progress} />}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto bg-gray-50">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500 p-8">
            <Database size={48} className="mb-4 text-gray-300" />
            <h2 className="text-xl font-medium mb-2">Start a conversation</h2>
            <p className="text-center text-sm max-w-md">
              Ask questions about your data and I'll search the knowledge base and execute
              analysis code to find answers.
            </p>

            {/* File upload */}
            <div className="mt-6 w-full max-w-md">
              <FileUploadZone
                onFileSelect={handleFileSelect}
                currentFile={dataFile}
                uploadProgress={uploadProgress}
                isUploading={isUploading}
                uploadSuccess={uploadSuccess}
              />
              {dataPath && (
                <p className="text-xs text-green-600 mt-2">✓ Data file ready for analysis</p>
              )}
            </div>

            {/* Options info */}
            <div className="mt-4 flex gap-4 text-xs text-gray-400">
              <span className={ragEnabled ? 'text-green-600' : ''}>
                RAG: {ragEnabled ? 'ON' : 'OFF'}
              </span>
              <span className={webSearchEnabled ? 'text-blue-600' : ''}>
                Web Search: {webSearchEnabled ? 'ON' : 'OFF'}
              </span>
            </div>
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {messages.map((msg) => (
              <MessageRenderer key={msg.id} message={msg} />
            ))}

            {/* Streaming indicator */}
            {isRunning && (
              <div className="flex gap-4 p-4 bg-white">
                <div className="w-8 h-8 rounded-lg gradient-bg text-white flex items-center justify-center">
                  <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full" />
                </div>
                <div className="flex items-center text-gray-500 text-sm">
                  {currentStage ? `Running: ${currentStage}...` : 'Processing...'}
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <ChatInput
        onSend={handleSend}
        isLoading={isRunning}
        disabled={!isConnected}
        placeholder={
          !isConnected
            ? 'Reconnecting...'
            : isRunning
              ? 'Waiting for response...'
              : dataPath
                ? `Ask about ${dataFile?.name}...`
                : 'Ask about your data...'
        }
      />
    </div>
  )
}

/** Render the appropriate component based on message type */
function MessageRenderer({ message }: { message: ChatMessageType }) {
  switch (message.type) {
    case 'user':
      return <ChatMessage role="user" content={message.content || ''} />

    case 'assistant':
      return <ChatMessage role="assistant" content={message.content || ''} />

    case 'rag_search':
      if (!message.ragGroup) return null
      return <RAGAgentMessage group={message.ragGroup} defaultExpanded={false} />

    case 'system':
      return (
        <div className="px-6 py-2 text-center text-sm text-gray-500 bg-gray-50">
          {message.content}
        </div>
      )

    default:
      return null
  }
}

export default WorkflowChatPage