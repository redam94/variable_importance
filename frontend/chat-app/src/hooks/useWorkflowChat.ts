import { useCallback, useState, useRef, useEffect } from 'react'
import { useWebSocket } from './useWebSocket'
import { getStoredToken } from '../lib/api'
import type { WSMessage, RAGSearchGroup, ChatMessage } from '../types/ws'
import { updateRAGGroups } from '../types/ws'

interface UseWorkflowChatOptions {
  workflowId?: string
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
  
  // Actions
  startWorkflow: (query: string) => Promise<string>
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
  
  // Generate stable workflowId if not provided
  const effectiveWorkflowId = useRef(
    workflowId || `workflow_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
  ).current
  
  // Track RAG groups by event_id
  const ragGroupsRef = useRef<Map<string, RAGSearchGroup>>(new Map())
  const messageIdCounter = useRef(0)
  
  const generateId = () => `msg_${++messageIdCounter.current}_${Date.now()}`

  const handleMessage = useCallback((msg: WSMessage) => {
    // Handle stage updates
    if (msg.stage) {
      setCurrentStage(msg.stage)
    }
    
    if (msg.data?.progress !== undefined) {
      setProgress(Number(msg.data.progress))
    }

    // Handle RAG events - update groups and messages
    if (msg.type === 'rag_search_start' || msg.type === 'rag_query' || msg.type === 'rag_search_end') {
      const eventId = msg.data?.event_id
      if (!eventId) return
      
      // Update the RAG group
      ragGroupsRef.current = updateRAGGroups(ragGroupsRef.current, msg)
      const group = ragGroupsRef.current.get(eventId)
      
      if (!group) return
      
      setMessages(prev => {
        // Find existing RAG message for this event
        const existingIdx = prev.findIndex(
          m => m.type === 'rag_search' && m.ragGroup?.event_id === eventId
        )
        
        const ragMessage: ChatMessage = {
          id: existingIdx >= 0 ? prev[existingIdx].id : generateId(),
          type: 'rag_search',
          ragGroup: { ...group },
          timestamp: group.timestamp,
        }
        
        if (existingIdx >= 0) {
          // Update existing
          const updated = [...prev]
          updated[existingIdx] = ragMessage
          return updated
        }
        
        // Add new (after last user message, before assistant response)
        return [...prev, ragMessage]
      })
    }

    // Handle completion
    if (msg.type === 'done') {
      setIsRunning(false)
      setProgress(100)
      onComplete?.(msg.task_id || '')
    }

    // Handle errors
    if (msg.type === 'error') {
      setIsRunning(false)
      onError?.(msg.message || 'Unknown error')
    }
  }, [onComplete, onError])

  const { isConnected, reconnect } = useWebSocket({
    workflowId: effectiveWorkflowId,
    onMessage: handleMessage,
  })

  const startWorkflow = useCallback(async (query: string): Promise<string> => {
    // Add user message
    const userMessage: ChatMessage = {
      id: generateId(),
      type: 'user',
      role: 'user',
      content: query,
      timestamp: new Date().toISOString(),
    }
    setMessages(prev => [...prev, userMessage])
    
    // Reset state
    ragGroupsRef.current.clear()
    setIsRunning(true)
    setProgress(0)
    setCurrentStage(null)
    
    // Call async endpoint with auth
    const token = getStoredToken()
    const response = await fetch('/api/workflow/run-async', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({
        query,
        workflow_id: effectiveWorkflowId,
      }),
    })
    
    if (!response.ok) {
      const error = await response.text()
      setIsRunning(false)
      throw new Error(error)
    }
    
    const { task_id } = await response.json()
    return task_id
  }, [effectiveWorkflowId])

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
    workflowId: effectiveWorkflowId,
    startWorkflow,
    clearMessages,
    reconnect,
  }
}