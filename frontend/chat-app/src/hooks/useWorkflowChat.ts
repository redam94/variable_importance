import { useCallback, useState, useRef } from 'react'
import { useWebSocket } from './useWebSocket'
import { getStoredToken } from '../lib/api'
import type { WSMessage, RAGSearchGroup, ChatMessage } from '../types/ws'
import { updateRAGGroups } from '../types/ws'

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

interface UseWorkflowChatReturn {
  isConnected: boolean
  isRunning: boolean
  messages: ChatMessage[]
  currentStage: string | null
  progress: number
  workflowId: string

  startWorkflow: (query: string, options?: WorkflowOptions) => Promise<string>
  clearMessages: () => void
  reconnect: () => void
}

export function useWorkflowChat({
  workflowId,
  onComplete,
  onError,
}: UseWorkflowChatOptions): UseWorkflowChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [currentStage, setCurrentStage] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  const [isRunning, setIsRunning] = useState(false)

  const ragGroupsRef = useRef<Map<string, RAGSearchGroup>>(new Map())
  const messageIdCounter = useRef(0)

  const generateId = () => `msg_${++messageIdCounter.current}_${Date.now()}`

  const handleMessage = useCallback(
    (msg: WSMessage) => {
      if (msg.stage) {
        setCurrentStage(msg.stage)
      }

      if (msg.data?.progress !== undefined) {
        setProgress(Number(msg.data.progress))
      }

      // Handle RAG events
      if (
        msg.type === 'rag_search_start' ||
        msg.type === 'rag_query' ||
        msg.type === 'rag_search_end'
      ) {
        const eventId = msg.data?.event_id
        if (!eventId) return

        ragGroupsRef.current = updateRAGGroups(ragGroupsRef.current, msg)
        const group = ragGroupsRef.current.get(eventId)

        if (!group) return

        setMessages((prev) => {
          const existingIdx = prev.findIndex(
            (m) => m.type === 'rag_search' && m.ragGroup?.event_id === eventId
          )

          const ragMessage: ChatMessage = {
            id: existingIdx >= 0 ? prev[existingIdx].id : generateId(),
            type: 'rag_search',
            ragGroup: { ...group },
            timestamp: group.timestamp,
          }

          if (existingIdx >= 0) {
            const updated = [...prev]
            updated[existingIdx] = ragMessage
            return updated
          }

          return [...prev, ragMessage]
        })
      }

      // Handle done with summary
      if (msg.type === 'done') {
        setIsRunning(false)
        setProgress(100)

        // Add assistant message with summary if present
        const summary = (msg as any).summary
        if (summary) {
          setMessages((prev) => [
            ...prev,
            {
              id: generateId(),
              type: 'assistant',
              role: 'assistant',
              content: summary,
              timestamp: new Date().toISOString(),
            },
          ])
        }

        onComplete?.(msg.task_id || '')
      }

      if (msg.type === 'error') {
        setIsRunning(false)
        onError?.(msg.message || 'Unknown error')
      }
    },
    [onComplete, onError]
  )

  const { isConnected, reconnect } = useWebSocket({
    workflowId,
    onMessage: handleMessage,
  })

  const startWorkflow = useCallback(
    async (query: string, options?: WorkflowOptions): Promise<string> => {
      const userMessage: ChatMessage = {
        id: generateId(),
        type: 'user',
        role: 'user',
        content: query,
        timestamp: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, userMessage])

      ragGroupsRef.current.clear()
      setIsRunning(true)
      setProgress(0)
      setCurrentStage(null)

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
          data_path: options?.dataPath || undefined,
          rag_enabled: options?.ragEnabled ?? true,
          web_search_enabled: options?.webSearchEnabled ?? false,
        }),
      })

      if (!response.ok) {
        const error = await response.text()
        setIsRunning(false)
        throw new Error(error)
      }

      const { task_id } = await response.json()
      return task_id
    },
    [workflowId]
  )

  const clearMessages = useCallback(() => {
    setMessages([])
    ragGroupsRef.current.clear()
    setCurrentStage(null)
    setProgress(0)
    setIsRunning(false)
  }, [])

  return {
    isConnected,
    isRunning,
    messages,
    currentStage,
    progress,
    workflowId,
    startWorkflow,
    clearMessages,
    reconnect,
  }
}