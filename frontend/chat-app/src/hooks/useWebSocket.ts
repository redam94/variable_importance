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
      console.log('[useWebSocket] Already connected/connecting, skipping')
      return
    }

    const token = getStoredToken()
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/workflow/${workflowId}${token ? `?token=${token}` : ''}`

    console.log('[useWebSocket] 🔌 Connecting to:', wsUrl)

    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('[useWebSocket] ✅ WebSocket opened')
      setIsConnected(true)
      onConnectRef.current?.()
    }

    ws.onmessage = (event) => {
      console.log('[useWebSocket] 📨 Raw message received:', event.data)
      try {
        const message: WSMessage = JSON.parse(event.data)
        console.log('[useWebSocket] 📨 Parsed message:', message.type, message)
        onMessageRef.current?.(message)
      } catch (e) {
        console.error('[useWebSocket] ❌ Failed to parse message:', event.data, e)
      }
    }

    ws.onclose = (event) => {
      console.log('[useWebSocket] 🔌 WebSocket closed:', event.code, event.reason)
      setIsConnected(false)
      wsRef.current = null
      onDisconnectRef.current?.()

      if (autoReconnect) {
        console.log('[useWebSocket] 🔄 Will reconnect in', reconnectInterval, 'ms')
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connect()
        }, reconnectInterval)
      }
    }

    ws.onerror = (error) => {
      console.error('[useWebSocket] ❌ WebSocket error:', error)
      onErrorRef.current?.(error)
    }
  }, [workflowId, autoReconnect, reconnectInterval])

  const disconnect = useCallback(() => {
    console.log('[useWebSocket] 🔌 Disconnecting...')
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
      console.log('[useWebSocket] 📤 Sending:', data)
      wsRef.current.send(JSON.stringify(data))
    } else {
      console.warn('[useWebSocket] ⚠️ Cannot send, WebSocket not open')
    }
  }, [])

  const reconnect = useCallback(() => {
    console.log('[useWebSocket] 🔄 Manual reconnect requested')
    disconnect()
    setTimeout(() => connect(), 100)
  }, [disconnect, connect])

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    console.log('[useWebSocket] 🚀 Hook mounted, connecting...')
    connect()

    // Ping to keep connection alive
    const pingInterval = setInterval(() => {
      send({ type: 'ping' })
    }, 30000)

    return () => {
      console.log('[useWebSocket] 🛑 Hook unmounting, disconnecting...')
      clearInterval(pingInterval)
      disconnect()
    }
  }, [connect, disconnect, send])

  // Handle workflowId changes by subscribing to new workflow
  useEffect(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      console.log('[useWebSocket] 📝 Subscribing to workflow:', workflowId)
      send({ type: 'subscribe', workflow_id: workflowId })
    }
  }, [workflowId, send])

  return { isConnected, send, disconnect, reconnect }
}