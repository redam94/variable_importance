import { useRef, useEffect, useCallback } from 'react'
import { useChatStore } from '../stores/chatStore'
import { useChatWebSocket } from '../hooks/useWebSocket'
import { chatApi } from '../lib/api'
import { ChatMessage } from '../components/chat/ChatMessage'
import { ChatInput } from '../components/chat/ChatInput'
import {
  Trash2,
  Settings,
  Wifi,
  WifiOff,
  RefreshCw,
  Copy,
  Check,
} from 'lucide-react'
import { useState } from 'react'
import clsx from 'clsx'

export function ChatPage() {
  const {
    workflowId,
    messages,
    isStreaming,
    streamingContent,
    error,
    addMessage,
    appendToStream,
    startStreaming,
    stopStreaming,
    clearMessages,
    setError,
    setWorkflowId,
  } = useChatStore()
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const [showSettings, setShowSettings] = useState(false)
  const [copied, setCopied] = useState(false)
  
  const { isConnected, currentStage, progress } = useChatWebSocket(workflowId)
  
  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingContent])
  
  const handleSend = useCallback(
    async (message: string) => {
      // Add user message
      addMessage({ role: 'user', content: message })
      startStreaming()
      setError(null)
      
      // Stream response
      abortControllerRef.current = chatApi.streamChat(
        {
          workflow_id: workflowId,
          message,
          history: messages,
        },
        (chunk) => {
          if (chunk.type === 'token') {
            appendToStream(chunk.content)
          } else if (chunk.type === 'done') {
            stopStreaming()
          } else if (chunk.type === 'error') {
            setError(chunk.content)
            stopStreaming()
          }
        },
        (error) => {
          setError(error.message)
          stopStreaming()
        }
      )
    },
    [workflowId, messages, addMessage, startStreaming, appendToStream, stopStreaming, setError]
  )
  
  const handleStop = useCallback(() => {
    abortControllerRef.current?.abort()
    stopStreaming()
  }, [stopStreaming])
  
  const handleNewWorkflow = () => {
    const now = new Date()
    const newId = `workflow_${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}`
    setWorkflowId(newId)
    clearMessages()
  }
  
  const handleCopyWorkflowId = () => {
    navigator.clipboard.writeText(workflowId)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }
  
  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-white">
        <div>
          <h1 className="text-xl font-semibold text-gray-900">💬 Chat</h1>
          <p className="text-sm text-gray-500">Stream responses with RAG context</p>
        </div>
        
        <div className="flex items-center gap-3">
          {/* Connection status */}
          <div
            className={clsx(
              'flex items-center gap-2 px-3 py-1.5 rounded-full text-sm',
              isConnected ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'
            )}
          >
            {isConnected ? <Wifi size={14} /> : <WifiOff size={14} />}
            {isConnected ? 'Connected' : 'Disconnected'}
          </div>
          
          {/* Progress indicator */}
          {currentStage && (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-primary-100 text-primary-700 rounded-full text-sm">
              <RefreshCw size={14} className="animate-spin" />
              {currentStage} ({Math.round(progress)}%)
            </div>
          )}
          
          {/* Settings */}
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="btn-ghost"
            title="Settings"
          >
            <Settings size={20} />
          </button>
          
          {/* Clear */}
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
          <div className="max-w-4xl mx-auto">
            <h3 className="font-medium text-gray-900 mb-3">⚙️ Settings</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Workflow ID */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Workflow ID
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={workflowId}
                    onChange={(e) => setWorkflowId(e.target.value)}
                    className="input-field text-sm font-mono"
                  />
                  <button
                    onClick={handleCopyWorkflowId}
                    className="btn-secondary px-3"
                    title="Copy"
                  >
                    {copied ? <Check size={16} /> : <Copy size={16} />}
                  </button>
                </div>
              </div>
              
              {/* New workflow */}
              <div className="flex items-end">
                <button
                  onClick={handleNewWorkflow}
                  className="btn-primary"
                >
                  🔄 New Workflow
                </button>
              </div>
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
      
      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 && !isStreaming ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-4">
            <div className="w-20 h-20 gradient-bg rounded-2xl flex items-center justify-center text-white text-4xl mb-6">
              💬
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              Start a Conversation
            </h2>
            <p className="text-gray-500 max-w-md mb-6">
              Ask questions about your data, request analyses, or explore insights.
              Your chat history will be preserved for this workflow.
            </p>
            <div className="flex flex-wrap gap-2 justify-center max-w-xl">
              {[
                'Analyze the trends in my data',
                'Create a visualization of sales by region',
                'What patterns do you see?',
                'Summarize the key findings',
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
          <div className="max-w-4xl mx-auto">
            {messages.map((msg, idx) => (
              <ChatMessage key={idx} role={msg.role} content={msg.content} />
            ))}
            
            {isStreaming && (
              <ChatMessage
                role="assistant"
                content={streamingContent}
                isStreaming
              />
            )}
            
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>
      
      {/* Input */}
      <ChatInput
        onSend={handleSend}
        onStop={handleStop}
        isLoading={isStreaming}
        placeholder="Ask me anything about your data..."
      />
    </div>
  )
}