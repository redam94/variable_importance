// =============================================================================
// WEBSOCKET MESSAGE TYPES - Extended for RAG Agent Events
// =============================================================================

export type WSMessageType =
  | 'connected'
  | 'progress'
  | 'stage_start'
  | 'stage_end'
  | 'rag_search_start'
  | 'rag_query'
  | 'rag_search_end'
  | 'code_generated'
  | 'execution_start'
  | 'execution_result'
  | 'error'
  | 'done'
  | 'pong'

export interface WSMessage {
  type: WSMessageType
  task_id?: string
  event?: string
  stage?: string
  message?: string
  timestamp?: string
  authenticated?: boolean
  username?: string
  workflow_id?: string
  data?: WSMessageData
}

export interface WSMessageData {
  // RAG events
  event_id?: string
  query?: string
  iteration?: number
  chunks_found?: number
  relevance?: number
  total_iterations?: number
  total_chunks?: number
  final_relevance?: number
  accepted?: boolean
  
  // Code events
  code?: string
  line_count?: number
  
  // Execution events
  success?: boolean
  output?: string
  
  // Error
  error?: string
  
  // Progress
  progress?: number
  
  // Generic
  [key: string]: unknown
}

// =============================================================================
// RAG AGENT TYPES - For collapsible display
// =============================================================================

export interface RAGQuery {
  query: string
  iteration: number
  chunks_found: number
  relevance?: number
}

export interface RAGSearchGroup {
  event_id: string
  status: 'searching' | 'complete' | 'failed'
  total_iterations: number
  total_chunks: number
  final_relevance: number
  accepted: boolean
  queries: RAGQuery[]
  timestamp: string
}

// =============================================================================
// CHAT MESSAGE TYPES - Extended
// =============================================================================

export type ChatMessageType = 'user' | 'assistant' | 'rag_search' | 'system'

export interface ChatMessage {
  id: string
  type: ChatMessageType
  role?: 'user' | 'assistant'
  content?: string
  ragGroup?: RAGSearchGroup
  timestamp: string
}

// =============================================================================
// HELPER: Build RAG groups from WebSocket events
// =============================================================================

export function updateRAGGroups(
  groups: Map<string, RAGSearchGroup>,
  message: WSMessage
): Map<string, RAGSearchGroup> {
  const newGroups = new Map(groups)
  const eventId = message.data?.event_id

  if (!eventId) return newGroups

  switch (message.type) {
    case 'rag_search_start':
      newGroups.set(eventId, {
        event_id: eventId,
        status: 'searching',
        total_iterations: 0,
        total_chunks: 0,
        final_relevance: 0,
        accepted: false,
        queries: [],
        timestamp: message.timestamp || new Date().toISOString(),
      })
      break

    case 'rag_query':
      const group = newGroups.get(eventId)
      if (group) {
        group.queries.push({
          query: message.data?.query || '',
          iteration: message.data?.iteration || group.queries.length + 1,
          chunks_found: message.data?.chunks_found || 0,
          relevance: message.data?.relevance,
        })
        group.total_iterations = group.queries.length
      }
      break

    case 'rag_search_end':
      const endGroup = newGroups.get(eventId)
      if (endGroup) {
        endGroup.status = message.data?.accepted ? 'complete' : 'failed'
        endGroup.total_iterations = message.data?.total_iterations || endGroup.queries.length
        endGroup.total_chunks = message.data?.total_chunks || 0
        endGroup.final_relevance = message.data?.final_relevance || 0
        endGroup.accepted = message.data?.accepted || false
      }
      break
  }

  return newGroups
}