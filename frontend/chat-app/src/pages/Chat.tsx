import { useRef, useEffect, useState, useCallback } from 'react'
import { useWorkflowChat, type WorkflowOptions } from '../hooks/useWorkflowChat'
import { workflowApi } from '../lib/api'
import { ChatMessage } from '../components/chat/ChatMessage'
import { RAGAgentMessage } from '../components/chat/RAGAgentMessage'
import { ChatInput } from '../components/chat/ChatInput'
import { WorkflowProgressPanel } from '../components/workflow/WorkflowProgressPanel'
import { FileUploadZone } from '../components/workflow/FileUploadZone'
import { useWorkflowImages } from '../hooks/useWorkflowImages'
import { ImageViewer, ImageGrid } from '../components/images/ImageViewer'
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
  Image,
  FolderOpen,
} from 'lucide-react'
import clsx from 'clsx'
import type { ChatMessage as ChatMessageType } from '../types/ws'

export function WorkflowChatPage() {
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [showSettings, setShowSettings] = useState(false)

  // Image state
  const [showImages, setShowImages] = useState(false)
  const [viewerOpen, setViewerOpen] = useState(false)
  const [viewerIndex, setViewerIndex] = useState(0)

  // Use the workflow chat hook with page type for separate state
  const {
    workflowId,
    isConnected,
    isRunning,
    messages,
    wsMessages,
    currentStage,
    progress,
    startedAt,
    currentTaskId,
    startWorkflow,
    clearMessages,
    reconnect,
  } = useWorkflowChat({
    pageType: 'workflow-chat', // Unique key for this page's chat state
    onComplete: (taskId) => {
      console.log('[Chat] ✅ Workflow complete:', taskId)
    },
    onError: (error) => {
      console.error('[Chat] ❌ Workflow error:', error)
    },
  })

  // Fetch images
  const { images, totalImages } = useWorkflowImages({
    workflowId,
    autoFetch: true,
  })

  // File upload state
  const [dataFile, setDataFile] = useState<File | null>(null)
  const [dataPath, setDataPath] = useState<string | null>(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadSuccess, setUploadSuccess] = useState(false)

  // Workflow options
  const [ragEnabled, setRagEnabled] = useState(true)
  const [webSearchEnabled, setWebSearchEnabled] = useState(false)

  // Debug log messages changes
  useEffect(() => {
    console.log('[Chat] 📝 Messages updated:', messages?.length ?? 0, messages)
  }, [messages])

  // Auto-scroll on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = async (query: string) => {
    console.log('[Chat] 📤 Sending query:', query)
    try {
      const options: WorkflowOptions = {
        dataPath,
        ragEnabled,
        webSearchEnabled,
      }
      await startWorkflow(query, options)
    } catch (error) {
      console.error('[Chat] Failed to start workflow:', error)
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

    try {
      const result = await workflowApi.uploadData(file, (progress) => {
        setUploadProgress(progress)
      })
      setDataPath(result.file_path)
      setUploadSuccess(true)
    } catch (error) {
      console.error('Upload failed:', error)
      setDataFile(null)
      setDataPath(null)
    } finally {
      setIsUploading(false)
    }
  }, [])

  const handleClearFile = useCallback(() => {
    setDataFile(null)
    setDataPath(null)
    setUploadSuccess(false)
    setUploadProgress(0)
  }, [])

  const handleClearAll = useCallback(() => {
    clearMessages()
    handleClearFile()
  }, [clearMessages, handleClearFile])

  return (
    <div className="flex flex-col h-full bg-white">
      {/* Header */}
      <div className="flex items-center justify-between px-6 py-3 border-b border-gray-200 bg-white">
        <div className="flex items-center gap-3">
          <div>
            <h1 className="text-lg font-semibold text-gray-900">Workflow Chat</h1>
            <div className="flex items-center gap-1 text-xs text-gray-500">
              <FolderOpen size={12} />
              <span>{workflowId || 'No workflow'}</span>
            </div>
          </div>
          {/* Connection status - only show during active workflow */}
          {isRunning ? (
            <div
              className={clsx(
                'flex items-center gap-1 px-2 py-0.5 rounded-full text-xs',
                isConnected ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'
              )}
            >
              {isConnected ? <Wifi size={12} /> : <WifiOff size={12} />}
              {isConnected ? 'Connected' : 'Connecting...'}
            </div>
          ) : (
            <div className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-gray-100 text-gray-600">
              Ready
            </div>
          )}
          {isRunning && currentTaskId && (
            <div className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs bg-blue-100 text-blue-700">
              <span className="w-2 h-2 bg-blue-500 rounded-full animate-pulse" />
              Task: {currentTaskId.slice(0, 8)}...
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          {/* Images button */}
          {totalImages > 0 && (
            <button
              onClick={() => setShowImages(!showImages)}
              className={clsx(
                'p-2 rounded-lg transition-colors flex items-center',
                showImages ? 'bg-purple-100 text-purple-700' : 'hover:bg-gray-100 text-gray-600'
              )}
              title={`View ${totalImages} images`}
            >
              <Image size={18} />
              <span className="ml-1 text-xs">{totalImages}</span>
            </button>
          )}

          {/* Only show reconnect when running but disconnected */}
          {isRunning && !isConnected && (
            <button
              onClick={reconnect}
              className="p-2 hover:bg-gray-100 rounded-lg text-gray-600"
              title="Reconnect"
            >
              <RefreshCw size={18} />
            </button>
          )}

          <button
            onClick={() => setShowSettings(!showSettings)}
            className={clsx(
              'p-2 rounded-lg transition-colors',
              showSettings ? 'bg-primary-100 text-primary-700' : 'hover:bg-gray-100 text-gray-600'
            )}
            title="Settings"
          >
            <Settings size={18} />
          </button>

          <button
            onClick={handleClearAll}
            className="p-2 hover:bg-red-50 text-red-600 rounded-lg"
            title="Clear chat"
          >
            <Trash2 size={18} />
          </button>
        </div>
      </div>

      {/* Image gallery (collapsible) */}
      {showImages && (images?.length ?? 0) > 0 && (
        <div className="border-b border-gray-200 p-4 bg-gray-50">
          <ImageGrid
            images={images}
            onImageClick={(idx) => {
              setViewerIndex(idx)
              setViewerOpen(true)
            }}
          />
        </div>
      )}

      {/* Full-screen viewer */}
      {viewerOpen && (
        <ImageViewer
          images={images}
          initialIndex={viewerIndex}
          onClose={() => setViewerOpen(false)}
        />
      )}

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

      {/* Progress panel - show when running or has recent activity */}
      {(isRunning || (wsMessages?.length ?? 0) > 0) && (
        <WorkflowProgressPanel
          stage={currentStage}
          progress={progress}
          isRunning={isRunning}
          wsMessages={wsMessages}
          startedAt={startedAt}
        />
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto bg-gray-50">
        {(!messages || messages.length === 0) ? (
          <div className="flex flex-col items-center justify-center h-full text-gray-500 p-8">
            {/* File upload zone when no messages */}
            {!dataPath ? (
              <div className="w-full max-w-md">
                <FileUploadZone
                  onFileSelect={handleFileSelect}
                  currentFile={dataFile}
                  uploadProgress={uploadProgress}
                  isUploading={isUploading}
                  uploadSuccess={uploadSuccess}
                  accept=".csv,.xlsx,.xls,.json,.parquet"
                />
                <p className="text-center text-sm mt-4 text-gray-400">
                  Or just start chatting - data file is optional
                </p>
              </div>
            ) : (
              <>
                <Database size={48} className="mb-4 text-gray-300" />
                <h2 className="text-xl font-medium mb-2">Data loaded: {dataFile?.name}</h2>
                <p className="text-center text-sm max-w-md">
                  Ask questions about your data and I'll search the knowledge base and execute
                  analysis code to find answers.
                </p>
              </>
            )}
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {(messages || []).map((message) => (
              <MessageRenderer key={message.id} message={message} />
            ))}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <ChatInput
        onSend={handleSend}
        isLoading={isRunning}
        disabled={false}
        placeholder={
          isRunning
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