import { useEffect, useRef, useCallback, useState } from 'react'
import type { WSMessage } from '../types/api'
import { getStoredToken } from '../lib/api'

interface UseWebSocketOptions {
  workflowId: string
  onMessage?: (message: WSMessage) => void
  onConnect?: () => void
  onDisconnect?: () => void
  onError?: (error: Event) => void
  autoReconnect?: boolean
  reconnectInterval?: number
}

interface UseWebSocketReturn {
  isConnected: boolean
  send: (data: Record<string, unknown>) => void
  disconnect: () => void
  reconnect: () => void
}

export function useWebSocket(options: UseWebSocketOptions): UseWebSocketReturn {
  const {
    workflowId,
    onMessage,
    onConnect,
    onDisconnect,
    onError,
    autoReconnect = true,
    reconnectInterval = 3000,
  } = options
  
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<number | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    
    const token = getStoredToken()
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/workflow/${workflowId}${token ? `?token=${token}` : ''}`
    
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws
    
    ws.onopen = () => {
      setIsConnected(true)
      onConnect?.()
    }
    
    ws.onmessage = (event) => {
      try {
        const message: WSMessage = JSON.parse(event.data)
        onMessage?.(message)
      } catch {
        console.error('Failed to parse WebSocket message:', event.data)
      }
    }
    
    ws.onclose = () => {
      setIsConnected(false)
      onDisconnect?.()
      
      if (autoReconnect) {
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect()
        }, reconnectInterval)
      }
    }
    
    ws.onerror = (error) => {
      onError?.(error)
    }
  }, [workflowId, onMessage, onConnect, onDisconnect, onError, autoReconnect, reconnectInterval])
  
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    
    setIsConnected(false)
  }, [])
  
  const send = useCallback((data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])
  
  const reconnect = useCallback(() => {
    disconnect()
    connect()
  }, [disconnect, connect])
  
  // Connect on mount, disconnect on unmount
  useEffect(() => {
    connect()
    
    // Ping to keep connection alive
    const pingInterval = setInterval(() => {
      send({ type: 'ping' })
    }, 30000)
    
    return () => {
      clearInterval(pingInterval)
      disconnect()
    }
  }, [connect, disconnect, send])
  
  // Reconnect when workflowId changes
  useEffect(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      send({ type: 'subscribe', workflow_id: workflowId })
    }
  }, [workflowId, send])
  
  return { isConnected, send, disconnect, reconnect }
}

// Hook for chat-specific WebSocket
export function useChatWebSocket(workflowId: string) {
  const [messages, setMessages] = useState<WSMessage[]>([])
  const [currentStage, setCurrentStage] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)
  
  const { isConnected, send, reconnect } = useWebSocket({
    workflowId,
    onMessage: (message) => {
      setMessages((prev) => [...prev.slice(-50), message]) // Keep last 50 messages
      
      if (message.stage) {
        setCurrentStage(message.stage)
      }
      
      if (message.type === 'progress' && message.data?.progress !== undefined) {
        setProgress(Number(message.data.progress))
      }
      
      if (message.type === 'done') {
        setProgress(100)
      }
    },
  })
  
  const clearMessages = useCallback(() => {
    setMessages([])
    setCurrentStage(null)
    setProgress(0)
  }, [])
  
  return {
    isConnected,
    messages,
    currentStage,
    progress,
    send,
    reconnect,
    clearMessages,
  }
}