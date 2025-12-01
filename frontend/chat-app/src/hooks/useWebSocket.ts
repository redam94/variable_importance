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

  // Store callbacks in refs to avoid reconnection on callback changes
  const onMessageRef = useRef(onMessage)
  const onConnectRef = useRef(onConnect)
  const onDisconnectRef = useRef(onDisconnect)
  const onErrorRef = useRef(onError)

  // Update refs when callbacks change (no reconnection triggered)
  useEffect(() => {
    onMessageRef.current = onMessage
  }, [onMessage])

  useEffect(() => {
    onConnectRef.current = onConnect
  }, [onConnect])

  useEffect(() => {
    onDisconnectRef.current = onDisconnect
  }, [onDisconnect])

  useEffect(() => {
    onErrorRef.current = onError
  }, [onError])

  const connect = useCallback(() => {
    // Don't connect if already connected or connecting
    if (wsRef.current?.readyState === WebSocket.OPEN ||
        wsRef.current?.readyState === WebSocket.CONNECTING) {
      return
    }

    const token = getStoredToken()
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/workflow/${workflowId}${token ? `?token=${token}` : ''}`

    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      setIsConnected(true)
      onConnectRef.current?.()
    }

    ws.onmessage = (event) => {
      try {
        const message: WSMessage = JSON.parse(event.data)
        onMessageRef.current?.(message)
      } catch {
        console.error('Failed to parse WebSocket message:', event.data)
      }
    }

    ws.onclose = () => {
      setIsConnected(false)
      wsRef.current = null
      onDisconnectRef.current?.()

      if (autoReconnect) {
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect()
        }, reconnectInterval)
      }
    }

    ws.onerror = (error) => {
      onErrorRef.current?.(error)
    }
  }, [workflowId, autoReconnect, reconnectInterval]) // Only reconnect when these change

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
    // Small delay before reconnecting
    setTimeout(() => connect(), 100)
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

  // Handle workflowId changes by subscribing to new workflow
  useEffect(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      send({ type: 'subscribe', workflow_id: workflowId })
    }
  }, [workflowId, send])

  return { isConnected, send, disconnect, reconnect }
}

// Hook for chat-specific WebSocket with stable callbacks
export function useChatWebSocket(workflowId: string) {
  const [messages, setMessages] = useState<WSMessage[]>([])
  const [currentStage, setCurrentStage] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)

  // Use useCallback with empty deps - state updates use functional form
  const handleMessage = useCallback((message: WSMessage) => {
    setMessages((prev) => [...prev.slice(-50), message])

    if (message.stage) {
      setCurrentStage(message.stage)
    }

    if (message.type === 'progress' && message.data?.progress !== undefined) {
      setProgress(Number(message.data.progress))
    }

    if (message.type === 'done') {
      setProgress(100)
    }
  }, [])

  const { isConnected, send, reconnect } = useWebSocket({
    workflowId,
    onMessage: handleMessage,
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