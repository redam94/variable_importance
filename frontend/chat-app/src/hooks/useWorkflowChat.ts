import { useCallback, useState, useRef, useEffect } from 'react'
import { getStoredToken } from '../lib/api'
import type { WSMessage, RAGSearchGroup, ChatMessage, RAGQuery } from '../types/ws'

// =============================================================================
// TYPES
// =============================================================================

export interface WorkflowOptions {
  dataPath?: string | null
  ragEnabled?: boolean
  webSearchEnabled?: boolean
}

interface UseWorkflowChatOptions {
  workflowId: string
  onComplete?: (taskId: string) => void
  onError?: (error: string) => void
}

// =============================================================================
// HELPERS
// =============================================================================

let messageIdCounter = 0
function generateId(): string {
  return `msg_${++messageIdCounter}_${Date.now()}`
}

// =============================================================================
// HOOK
// =============================================================================

export function useWorkflowChat(options: UseWorkflowChatOptions) {
  const { workflowId, onComplete, onError } = options

  // Local state for messages and progress
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [wsMessages, setWsMessages] = useState<WSMessage[]>([])
  const [currentStage, setCurrentStage] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [isRunning, setIsRunning] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [startedAt, setStartedAt] = useState<string | null>(null)
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null)

  // WebSocket ref
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<number | null>(null)

  // RAG groups tracking
  const ragGroupsRef = useRef<Map<string, RAGSearchGroup>>(new Map())

  // Store callbacks in refs to avoid stale closures
  const onCompleteRef = useRef(onComplete)
  const onErrorRef = useRef(onError)

  useEffect(() => {
    onCompleteRef.current = onComplete
  }, [onComplete])

  useEffect(() => {
    onErrorRef.current = onError
  }, [onError])

  // Handle incoming websocket messages
  const handleMessage = useCallback((msg: WSMessage) => {
    console.log('[WS] 📨 Received message:', msg.type, msg)

    // Store all WS messages for activity log
    setWsMessages(prev => [...prev.slice(-49), msg])

    // Update stage from message
    if (msg.stage) {
      console.log('[WS] 📍 Stage update:', msg.stage)
      setCurrentStage(msg.stage)
    }

    // Update progress from message data
    if (msg.data?.progress !== undefined) {
      console.log('[WS] 📊 Progress update:', msg.data.progress)
      setProgress(Number(msg.data.progress))
    }

    // Handle RAG events - group them
    if (msg.type === 'rag_search_start' || msg.type === 'rag_query' || msg.type === 'rag_search_end') {
      const eventId = msg.data?.event_id || 'default'
      console.log('[WS] 🔍 RAG event:', msg.type, 'eventId:', eventId)

      if (msg.type === 'rag_search_start') {
        ragGroupsRef.current.set(eventId, {
          event_id: eventId,
          status: 'searching',
          total_iterations: 0,
          total_chunks: 0,
          final_relevance: 0,
          accepted: false,
          queries: [],
          timestamp: msg.timestamp || new Date().toISOString(),
        })
      } else if (msg.type === 'rag_query') {
        const existing = ragGroupsRef.current.get(eventId)
        if (existing) {
          const query: RAGQuery = {
            query: msg.data?.query || '',
            iteration: msg.data?.iteration || 0,
            chunks_found: msg.data?.chunks_found || 0,
            relevance: msg.data?.relevance,
          }
          existing.queries.push(query)
          existing.total_iterations = msg.data?.iteration || existing.total_iterations
        }
      } else if (msg.type === 'rag_search_end') {
        const existing = ragGroupsRef.current.get(eventId)
        if (existing) {
          existing.status = 'complete'
          existing.total_iterations = msg.data?.total_iterations || existing.total_iterations
          existing.total_chunks = msg.data?.total_chunks || 0
          existing.final_relevance = msg.data?.final_relevance || 0
          existing.accepted = msg.data?.accepted || false

          // Add RAG message to chat when search completes
          const ragMessage: ChatMessage = {
            id: generateId(),
            type: 'rag_search',
            ragGroup: { ...existing },
            timestamp: existing.timestamp,
          }
          console.log('[WS] ➕ Adding RAG message:', ragMessage)
          setMessages(prev => [...prev, ragMessage])
        }
      }
    }

    // Handle done with summary
    if (msg.type === 'done') {
      console.log('[WS] ✅ DONE received!')
      console.log('[WS] ✅ Full message:', JSON.stringify(msg, null, 2))

      setIsRunning(false)
      setProgress(100)
      setCurrentStage('complete')

      // Get summary from various possible locations
      const summary = msg.summary || msg.data?.summary || msg.data?.result_summary || msg.result_summary
      console.log('[WS] ✅ Summary:', summary)

      if (summary) {
        const assistantMessage: ChatMessage = {
          id: generateId(),
          type: 'assistant',
          role: 'assistant',
          content: summary,
          timestamp: new Date().toISOString(),
        }
        console.log('[WS] ➕ Adding assistant message:', assistantMessage)
        setMessages(prev => [...prev, assistantMessage])
      } else {
        console.log('[WS] ⚠️ No summary in done message!')
      }

      // Close WebSocket after done
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }

      onCompleteRef.current?.(msg.task_id || '')
    }

    // Handle errors
    if (msg.type === 'error') {
      console.log('[WS] ❌ Error received:', msg.message)
      setIsRunning(false)
      
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
      
      onErrorRef.current?.(msg.message || 'Unknown error')
    }
  }, [])

  // Connect to task WebSocket
  const connectToTask = useCallback((taskId: string) => {
    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }

    const token = getStoredToken()
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/task/${taskId}${token ? `?token=${token}` : ''}`

    console.log('[WS] 🔌 Connecting to task WebSocket:', wsUrl)

    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('[WS] ✅ Task WebSocket opened')
      setIsConnected(true)
    }

    ws.onmessage = (event) => {
      console.log('[WS] 📨 Raw message received:', event.data)
      try {
        const message: WSMessage = JSON.parse(event.data)
        handleMessage(message)
      } catch (e) {
        console.error('[WS] ❌ Failed to parse message:', event.data, e)
      }
    }

    ws.onclose = (event) => {
      console.log('[WS] 🔌 Task WebSocket closed:', event.code, event.reason)
      setIsConnected(false)
      wsRef.current = null
    }

    ws.onerror = (error) => {
      console.error('[WS] ❌ WebSocket error:', error)
    }
  }, [handleMessage])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [])

  const startWorkflow = useCallback(
    async (query: string, workflowOptions?: WorkflowOptions): Promise<string> => {
      console.log('[Workflow] 🚀 Starting workflow with query:', query)

      // Add user message
      const userMessage: ChatMessage = {
        id: generateId(),
        type: 'user',
        role: 'user',
        content: query,
        timestamp: new Date().toISOString(),
      }
      setMessages(prev => [...prev, userMessage])

      // Clear RAG groups and reset state
      ragGroupsRef.current.clear()
      setWsMessages([])
      setIsRunning(true)
      setProgress(0)
      setCurrentStage(null)
      setStartedAt(new Date().toISOString())

      try {
        const token = getStoredToken()
        console.log('[Workflow] 📤 Sending request to /api/workflow/run-async')

        const response = await fetch('/api/workflow/run-async', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
          body: JSON.stringify({
            query,
            workflow_id: workflowId,
            data_path: workflowOptions?.dataPath || undefined,
            rag_enabled: workflowOptions?.ragEnabled ?? true,
            web_search_enabled: workflowOptions?.webSearchEnabled ?? false,
          }),
        })

        if (!response.ok) {
          const error = await response.text()
          console.log('[Workflow] ❌ Request failed:', error)
          setIsRunning(false)
          throw new Error(error)
        }

        const data = await response.json()
        console.log('[Workflow] 📥 Response:', data)

        const taskId = data.task_id
        setCurrentTaskId(taskId)

        // Connect to task-specific WebSocket for progress updates
        console.log('[Workflow] 🔌 Connecting to task WebSocket for:', taskId)
        connectToTask(taskId)

        return taskId
      } catch (error) {
        console.error('[Workflow] ❌ Error:', error)
        setIsRunning(false)
        throw error
      }
    },
    [workflowId, connectToTask]
  )

  const clearMessages = useCallback(() => {
    console.log('[Workflow] 🧹 Clearing messages')
    setMessages([])
    setWsMessages([])
    ragGroupsRef.current.clear()
    setCurrentStage(null)
    setProgress(0)
    setIsRunning(false)
    setStartedAt(null)
    setCurrentTaskId(null)
    
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
  }, [])

  const reconnect = useCallback(() => {
    if (currentTaskId) {
      connectToTask(currentTaskId)
    }
  }, [currentTaskId, connectToTask])

  return {
    // Connection state
    isConnected,
    reconnect,

    // Workflow state
    isRunning,
    currentStage,
    progress,
    startedAt,
    currentTaskId,

    // Messages
    messages,
    wsMessages,

    // Actions
    startWorkflow,
    clearMessages,
    workflowId,
  }
}