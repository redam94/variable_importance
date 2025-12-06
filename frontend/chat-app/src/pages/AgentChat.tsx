import { useRef, useEffect, useCallback, useState } from 'react'
import { useChatStore, selectMessages, selectCurrentWorkflowId } from '../stores/chatStore'
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
  FolderOpen,
} from 'lucide-react'
import clsx from 'clsx'
import type { ChatMessage as ChatMessageType } from '../types/ws'

const PAGE_TYPE = 'agent-chat'

interface ToolEvent {
  type: 'tool_start' | 'tool_end'
  tool: string
  input?: string
  output?: string
}

export function AgentChatPage() {
  // Get messages from agent-chat session
  const messages = useChatStore(selectMessages(PAGE_TYPE))
  
  // Get SHARED workflowId from workflow-chat (consistent across all pages)
  const workflowId = useChatStore(selectCurrentWorkflowId)
  
  // Get actions
  const initSession = useChatStore((state) => state.initSession)
  const addMessage = useChatStore((state) => state.addMessage)
  const clearMessages = useChatStore((state) => state.clearMessages)
  
  // Local state for streaming (not persisted)
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingContent, setStreamingContent] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [toolEvents, setToolEvents] = useState<ToolEvent[]>([])
  const [showTools, setShowTools] = useState(false)
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const streamingContentRef = useRef('')
  const doneProcessedRef = useRef(false)  // Guard against duplicate 'done' events
  
  // Initialize sessions on mount
  useEffect(() => {
    // Init workflow-chat to ensure shared workflowId exists
    initSession('workflow-chat')
    // Init agent-chat for this page's messages
    initSession(PAGE_TYPE)
  }, [initSession])
  
  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingContent])
  
  const handleSend = useCallback(
    async (message: string) => {
      // Add user message to store
      const userMsg: ChatMessageType = {
        id: `msg_${Date.now()}`,
        type: 'user',
        role: 'user',
        content: message,
        timestamp: new Date().toISOString(),
      }
      addMessage(PAGE_TYPE, userMsg)
      
      setIsStreaming(true)
      setStreamingContent('')
      streamingContentRef.current = ''
      doneProcessedRef.current = false  // Reset guard for new request
      setError(null)
      setToolEvents([])
      
      // Convert messages to history format for API (handle undefined during hydration)
      const history = (messages || []).map((m) => ({
        role: m.role || (m.type === 'user' ? 'user' : 'assistant'),
        content: m.content || '',
      }))
      
      // Stream response using agent endpoint
      abortControllerRef.current = chatApi.streamAgentChat(
        {
          workflow_id: workflowId,
          message,
          history,
        },
        (chunk) => {
          if (chunk.type === 'token') {
            setStreamingContent((prev) => {
              const newContent = prev + chunk.content
              streamingContentRef.current = newContent
              return newContent
            })
          } else if (chunk.type === 'tool_start') {
            setToolEvents((prev) => [
              ...prev,
              { type: 'tool_start', tool: chunk.content },
            ])
          } else if (chunk.type === 'tool_end') {
            setToolEvents((prev) => {
              const events = [...prev]
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
            // Guard against duplicate done events
            if (doneProcessedRef.current) {
              console.log('[AgentChat] Ignoring duplicate done event')
              return
            }
            doneProcessedRef.current = true
            
            // Add assistant message to store - use ref to avoid closure issues
            const content = streamingContentRef.current
            if (content) {
              const assistantMsg: ChatMessageType = {
                id: `msg_${Date.now()}`,
                type: 'assistant',
                role: 'assistant',
                content,
                timestamp: new Date().toISOString(),
              }
              addMessage(PAGE_TYPE, assistantMsg)
            }
            setStreamingContent('')
            streamingContentRef.current = ''
            setIsStreaming(false)
          } else if (chunk.type === 'error') {
            setError(chunk.content)
            setIsStreaming(false)
          }
        },
        (error) => {
          setError(error.message)
          setIsStreaming(false)
        }
      )
    },
    [workflowId, messages, addMessage]
  )
  
  const handleStop = useCallback(() => {
    abortControllerRef.current?.abort()
    setIsStreaming(false)
  }, [])
  
  const handleClear = useCallback(() => {
    clearMessages(PAGE_TYPE)
    setStreamingContent('')
    streamingContentRef.current = ''
    doneProcessedRef.current = false
    setError(null)
    setToolEvents([])
  }, [clearMessages])
  
  return (
    <div className="flex flex-col h-full min-h-screen">
      {/* Header */}
      <header className="flex-shrink-0 flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-white">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 gradient-bg rounded-xl flex items-center justify-center text-white">
            <Bot size={22} />
          </div>
          <div>
            <h1 className="text-xl font-semibold text-gray-900">RAG Agent</h1>
            <div className="flex items-center gap-2 text-sm text-gray-500">
              <FolderOpen size={14} />
              <span>{workflowId || 'No workflow selected'}</span>
            </div>
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
            onClick={handleClear}
            className="btn-ghost text-red-500 hover:bg-red-50"
            title="Clear messages"
          >
            <Trash2 size={20} />
          </button>
        </div>
      </header>
      
      {/* Tool events panel */}
      {showTools && toolEvents.length > 0 && (
        <div className="flex-shrink-0 px-6 py-4 bg-gray-50 border-b border-gray-200 max-h-48 overflow-y-auto">
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
        <div className="flex-shrink-0 px-6 py-3 bg-red-50 border-b border-red-200 text-red-700 text-sm">
          <strong>Error:</strong> {error}
        </div>
      )}
      
      {/* Messages */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        {(!messages || messages.length === 0) && !isStreaming ? (
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
            {(messages || []).map((msg, idx) => (
              <ChatMessage 
                key={msg.id || idx} 
                role={msg.role || 'assistant'} 
                content={msg.content || ''} 
              />
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
      <div className="flex-shrink-0">
      <ChatInput
        onSend={handleSend}
        onStop={handleStop}
        isLoading={isStreaming}
        placeholder="Ask me anything... I'll search the knowledge base for context"
      />
      </div>
    </div>
  )
}