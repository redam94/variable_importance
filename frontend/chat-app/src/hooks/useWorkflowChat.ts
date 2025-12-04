import { useCallback, useRef, useEffect } from 'react'
import { getStoredToken } from '../lib/api'
import {
  useChatStore,
  selectMessages,
  selectWsMessages,
  selectIsRunning,
  selectWorkflowId,
  selectCurrentTaskId,
  selectCurrentStage,
  selectProgress,
  selectStartedAt,
  selectRagGroups,
} from '../stores/chatStore'
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
  pageType: string
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
  const { pageType, onComplete, onError } = options

  // Get state using selectors (prevents unnecessary re-renders)
  const messages = useChatStore(selectMessages(pageType))
  const wsMessages = useChatStore(selectWsMessages(pageType))
  const isRunning = useChatStore(selectIsRunning(pageType))
  const workflowId = useChatStore(selectWorkflowId(pageType))
  const currentTaskId = useChatStore(selectCurrentTaskId(pageType))
  const currentStage = useChatStore(selectCurrentStage(pageType))
  const progress = useChatStore(selectProgress(pageType))
  const startedAt = useChatStore(selectStartedAt(pageType))
  const ragGroups = useChatStore(selectRagGroups(pageType))

  // Get actions directly from store (stable references)
  const initSession = useChatStore((state) => state.initSession)
  const loadWorkflow = useChatStore((state) => state.loadWorkflow)
  const newSession = useChatStore((state) => state.newSession)
  const addMessage = useChatStore((state) => state.addMessage)
  const addWsMessage = useChatStore((state) => state.addWsMessage)
  const storeClearMessages = useChatStore((state) => state.clearMessages)
  const startWorkflowRun = useChatStore((state) => state.startWorkflowRun)
  const updateWorkflowProgress = useChatStore((state) => state.updateWorkflowProgress)
  const completeWorkflowRun = useChatStore((state) => state.completeWorkflowRun)
  const failWorkflowRun = useChatStore((state) => state.failWorkflowRun)
  const updateRagGroup = useChatStore((state) => state.updateRagGroup)

  // Initialize session on mount (only if doesn't exist)
  useEffect(() => {
    initSession(pageType)
  }, [pageType, initSession])

  // WebSocket ref
  const wsRef = useRef<WebSocket | null>(null)

  // Store callbacks in refs
  const onCompleteRef = useRef(onComplete)
  const onErrorRef = useRef(onError)
  const ragGroupsRef = useRef(ragGroups)

  useEffect(() => {
    onCompleteRef.current = onComplete
  }, [onComplete])

  useEffect(() => {
    onErrorRef.current = onError
  }, [onError])

  useEffect(() => {
    ragGroupsRef.current = ragGroups
  }, [ragGroups])

  // Handle incoming websocket messages
  const handleMessage = useCallback((msg: WSMessage) => {
    console.log('[WS] 📨 Received message:', msg.type, msg)

    addWsMessage(pageType, msg)

    if (msg.stage) {
      updateWorkflowProgress(pageType, { currentStage: msg.stage })
    }

    if (msg.data?.progress !== undefined) {
      updateWorkflowProgress(pageType, { progress: Number(msg.data.progress) })
    }

    // Handle RAG events
    if (msg.type === 'rag_search_start' || msg.type === 'rag_query' || msg.type === 'rag_search_end') {
      const eventId = msg.data?.event_id || 'default'

      if (msg.type === 'rag_search_start') {
        const group: RAGSearchGroup = {
          event_id: eventId,
          status: 'searching',
          total_iterations: 0,
          total_chunks: 0,
          final_relevance: 0,
          accepted: false,
          queries: [],
          timestamp: msg.timestamp || new Date().toISOString(),
        }
        updateRagGroup(pageType, eventId, group)
      } else if (msg.type === 'rag_query') {
        const existing = ragGroupsRef.current[eventId]
        if (existing) {
          const query: RAGQuery = {
            query: msg.data?.query || '',
            iteration: msg.data?.iteration || 0,
            chunks_found: msg.data?.chunks_found || 0,
            relevance: msg.data?.relevance,
          }
          updateRagGroup(pageType, eventId, {
            ...existing,
            queries: [...existing.queries, query],
            total_iterations: msg.data?.iteration || existing.total_iterations,
          })
        }
      } else if (msg.type === 'rag_search_end') {
        const existing = ragGroupsRef.current[eventId]
        if (existing) {
          const completed: RAGSearchGroup = {
            ...existing,
            status: 'complete',
            total_iterations: msg.data?.total_iterations || existing.total_iterations,
            total_chunks: msg.data?.total_chunks || 0,
            final_relevance: msg.data?.final_relevance || 0,
            accepted: msg.data?.accepted || false,
          }
          updateRagGroup(pageType, eventId, completed)

          const ragMessage: ChatMessage = {
            id: generateId(),
            type: 'rag_search',
            ragGroup: completed,
            timestamp: completed.timestamp,
          }
          addMessage(pageType, ragMessage)
        }
      }
    }

    // Handle done
    if (msg.type === 'done') {
      console.log('[WS] ✅ DONE received!')
      completeWorkflowRun(pageType)

      const summary = msg.summary || msg.data?.summary || msg.data?.result_summary || msg.result_summary
      if (summary) {
        const assistantMessage: ChatMessage = {
          id: generateId(),
          type: 'assistant',
          role: 'assistant',
          content: summary,
          timestamp: new Date().toISOString(),
        }
        addMessage(pageType, assistantMessage)
      }

      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }

      onCompleteRef.current?.(msg.task_id || '')
    }

    // Handle errors
    if (msg.type === 'error') {
      console.log('[WS] ❌ Error received:', msg.message)
      failWorkflowRun(pageType)

      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }

      onErrorRef.current?.(msg.message || 'Unknown error')
    }
  }, [pageType, addMessage, addWsMessage, updateWorkflowProgress, completeWorkflowRun, failWorkflowRun, updateRagGroup])

  // Connect to task WebSocket
  const connectToTask = useCallback((taskId: string) => {
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
    }

    ws.onmessage = (event) => {
      try {
        const message: WSMessage = JSON.parse(event.data)
        handleMessage(message)
      } catch (e) {
        console.error('[WS] ❌ Failed to parse message:', event.data, e)
      }
    }

    ws.onclose = (event) => {
      console.log('[WS] 🔌 Task WebSocket closed:', event.code, event.reason)
      wsRef.current = null
    }

    ws.onerror = (error) => {
      console.error('[WS] ❌ WebSocket error:', error)
    }
  }, [handleMessage])

  // Reconnect to existing task if running
  useEffect(() => {
    if (isRunning && currentTaskId && !wsRef.current) {
      console.log('[WS] 🔄 Reconnecting to existing task:', currentTaskId)
      connectToTask(currentTaskId)
    }

    return () => {
      if (!isRunning && wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [isRunning, currentTaskId, connectToTask])

  const startWorkflow = useCallback(
    async (query: string, workflowOptions?: WorkflowOptions): Promise<string> => {
      console.log('[Workflow] 🚀 Starting workflow with query:', query)

      const userMessage: ChatMessage = {
        id: generateId(),
        type: 'user',
        role: 'user',
        content: query,
        timestamp: new Date().toISOString(),
      }
      addMessage(pageType, userMessage)

      try {
        const token = getStoredToken()
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
          throw new Error(error)
        }

        const data = await response.json()
        const taskId = data.task_id

        startWorkflowRun(pageType, taskId)
        connectToTask(taskId)

        return taskId
      } catch (error) {
        console.error('[Workflow] ❌ Error:', error)
        failWorkflowRun(pageType)
        throw error
      }
    },
    [pageType, workflowId, connectToTask, addMessage, startWorkflowRun, failWorkflowRun]
  )

  const clearMessages = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    storeClearMessages(pageType)
  }, [pageType, storeClearMessages])

  const reconnect = useCallback(() => {
    if (currentTaskId) {
      connectToTask(currentTaskId)
    }
  }, [currentTaskId, connectToTask])

  // Load a different workflow (clears current session)
  const switchWorkflow = useCallback((newWorkflowId: string) => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    loadWorkflow(pageType, newWorkflowId)
  }, [pageType, loadWorkflow])

  // Start fresh session with new workflow ID
  const startNewSession = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    newSession(pageType)
  }, [pageType, newSession])

  const isConnected = wsRef.current?.readyState === WebSocket.OPEN

  return {
    workflowId,
    isConnected,
    reconnect,
    isRunning,
    currentStage,
    progress,
    startedAt,
    currentTaskId,
    messages,
    wsMessages,
    startWorkflow,
    clearMessages,
    switchWorkflow,
    startNewSession,
  }
}