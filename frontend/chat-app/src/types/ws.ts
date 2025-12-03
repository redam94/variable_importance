// =============================================================================
// WEBSOCKET MESSAGE TYPES - Extended for RAG Agent Events
// =============================================================================

export type WSMessageType =
  | 'connected'
  | 'subscribed'
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
  summary?: string
  action?: string
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
// CHAT MESSAGE TYPES - Extended for workflow chat
// =============================================================================

export type ChatMessageType = 'user' | 'assistant' | 'rag_search' | 'system'

export interface ChatMessage {
  id: string
  type: ChatMessageType
  role?: 'user' | 'assistant'
  content?: string
  ragGroup?: RAGSearchGroup
  timestamp?: string
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/**
 * Update RAG groups map with incoming websocket message
 */
export function updateRAGGroups(
  groups: Map<string, RAGSearchGroup>,
  msg: WSMessage
): Map<string, RAGSearchGroup> {
  const eventId = msg.data?.event_id
  if (!eventId) return groups

  const newGroups = new Map(groups)

  if (msg.type === 'rag_search_start') {
    newGroups.set(eventId, {
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
    const existing = newGroups.get(eventId)
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
    const existing = newGroups.get(eventId)
    if (existing) {
      existing.status = 'complete'
      existing.total_iterations = msg.data?.total_iterations || existing.total_iterations
      existing.total_chunks = msg.data?.total_chunks || 0
      existing.final_relevance = msg.data?.final_relevance || 0
      existing.accepted = msg.data?.accepted || false
    }
  }

  return newGroups
}