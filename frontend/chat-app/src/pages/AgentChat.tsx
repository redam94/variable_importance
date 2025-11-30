import { useRef, useEffect, useCallback, useState } from 'react'
import { useChatStore } from '../stores/chatStore'
import { chatApi } from '../lib/api'
import { ChatMessage } from '../components/chat/ChatMessage'
import { ChatInput } from '../components/chat/ChatInput'
import {
  Trash2,
  Bot,
  Search,
  BookOpen,
  Sparkles,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'
import clsx from 'clsx'

interface ToolEvent {
  type: 'tool_start' | 'tool_end'
  tool: string
  input?: string
  output?: string
}

export function AgentChatPage() {
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
  } = useChatStore()
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const [toolEvents, setToolEvents] = useState<ToolEvent[]>([])
  const [showTools, setShowTools] = useState(false)
  
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
      setToolEvents([])
      
      // Stream response using agent endpoint
      abortControllerRef.current = chatApi.streamAgentChat(
        {
          workflow_id: workflowId,
          message,
          history: messages,
        },
        (chunk) => {
          if (chunk.type === 'token') {
            appendToStream(chunk.content)
          } else if (chunk.type === 'tool_start') {
            setToolEvents((prev) => [
              ...prev,
              { type: 'tool_start', tool: chunk.content },
            ])
          } else if (chunk.type === 'tool_end') {
            setToolEvents((prev) => {
              const events = [...prev]
              // Find last tool_start from the end
              let lastStart = -1
              for (let i = events.length - 1; i >= 0; i--) {
                if (events[i].type === 'tool_start') {
                  lastStart = i
                  break
                }
              }
              if (lastStart >= 0) {
                events[lastStart] = {
                  ...events[lastStart],
                  type: 'tool_end',
                  output: chunk.content,
                }
              }
              return events
            })
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
  
  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-white">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 gradient-bg rounded-xl flex items-center justify-center text-white">
            <Bot size={22} />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-gray-900">RAG Agent</h1>
            <p className="text-sm text-gray-500">
              AI with knowledge retrieval and storage tools
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          {/* Tool events toggle */}
          {toolEvents.length > 0 && (
            <button
              onClick={() => setShowTools(!showTools)}
              className={clsx(
                'flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-colors',
                showTools
                  ? 'bg-primary-100 text-primary-700'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              )}
            >
              <Search size={14} />
              {toolEvents.length} tool calls
              {showTools ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
          )}
          
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
      
      {/* Tool events panel */}
      {showTools && toolEvents.length > 0 && (
        <div className="px-6 py-4 bg-gray-50 border-b border-gray-200 max-h-48 overflow-y-auto">
          <h3 className="text-sm font-medium text-gray-700 mb-2">
            🔧 Tool Calls
          </h3>
          <div className="space-y-2">
            {toolEvents.map((event, idx) => (
              <div
                key={idx}
                className={clsx(
                  'flex items-start gap-3 p-3 rounded-lg text-sm',
                  event.type === 'tool_end' ? 'bg-green-50' : 'bg-yellow-50'
                )}
              >
                <div
                  className={clsx(
                    'w-6 h-6 rounded flex items-center justify-center flex-shrink-0',
                    event.type === 'tool_end'
                      ? 'bg-green-100 text-green-600'
                      : 'bg-yellow-100 text-yellow-600'
                  )}
                >
                  {event.tool === 'retrieve_context' ? (
                    <Search size={14} />
                  ) : (
                    <BookOpen size={14} />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-gray-900">{event.tool}</p>
                  {event.output && (
                    <p className="text-gray-600 truncate mt-1">{event.output}</p>
                  )}
                </div>
              </div>
            ))}
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
            <div className="w-20 h-20 gradient-bg rounded-2xl flex items-center justify-center text-white mb-6">
              <Sparkles size={40} />
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              RAG Agent Chat
            </h2>
            <p className="text-gray-500 max-w-md mb-6">
              This agent can search your knowledge base to find relevant context
              and store new information for future reference.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-2xl">
              <div className="card p-4 text-left">
                <div className="flex items-center gap-2 text-primary-600 mb-2">
                  <Search size={18} />
                  <span className="font-medium">Retrieve Context</span>
                </div>
                <p className="text-sm text-gray-600">
                  The agent searches your uploaded documents and stored knowledge
                  to find relevant information.
                </p>
              </div>
              
              <div className="card p-4 text-left">
                <div className="flex items-center gap-2 text-primary-600 mb-2">
                  <BookOpen size={18} />
                  <span className="font-medium">Store Knowledge</span>
                </div>
                <p className="text-sm text-gray-600">
                  Share useful information with the agent and it will save it
                  for future reference.
                </p>
              </div>
            </div>
            
            <div className="flex flex-wrap gap-2 justify-center max-w-xl mt-6">
              {[
                'What do you know about my data?',
                'Find information about sales trends',
                'Remember that our fiscal year starts in April',
                'Search for customer feedback patterns',
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
        placeholder="Ask me anything... I'll search the knowledge base for context"
      />
    </div>
  )
}