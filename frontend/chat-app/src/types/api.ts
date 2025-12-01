// =============================================================================
// AUTH TYPES
// =============================================================================

export interface User {
  username: string
  email: string | null
  full_name: string | null
  disabled: boolean
  created_at: string
}

export interface Token {
  access_token: string
  token_type: string
  expires_in: number
  refresh_token?: string
}

export interface LoginRequest {
  username: string
  password: string
}

export interface RegisterRequest {
  username: string
  password: string
  email?: string
  full_name?: string
}

// =============================================================================
// WORKFLOW TYPES
// =============================================================================

export interface WorkflowRequest {
  query: string
  workflow_id?: string
  stage_name?: string
  model?: string
}

export interface WorkflowMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp?: string
}

export interface WorkflowResponse {
  workflow_id: string
  stage_name: string
  message: WorkflowMessage
  artifacts: string[]
  execution_time_seconds: number
}

export interface WorkflowStatus {
  task_id: string
  workflow_id: string
  stage_name: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  current_node?: string
  progress: number
  elapsed_seconds?: number
}

// =============================================================================
// CHAT TYPES
// =============================================================================

export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export interface ChatRequest {
  workflow_id: string
  message: string
  history?: ChatMessage[]
  model?: string
}

export interface StreamChunk {
  type: 'start' | 'token' | 'done' | 'error'
  content: string
}

// =============================================================================
// DOCUMENT TYPES
// =============================================================================

export type DocumentSourceType = 'pdf' | 'txt' | 'md' | 'csv' | 'json' | 'url'

export interface DocumentUploadResponse {
  success: boolean
  title: string
  source_type: DocumentSourceType
  chunk_count: number
  content_length: number
  workflow_id: string
  message: string
}

export interface URLScrapeRequest {
  url: string
  workflow_id: string
  title?: string
}

export interface URLScrapeResponse {
  success: boolean
  url: string
  title: string
  chunk_count: number
  content_length: number
  workflow_id: string
  message: string
}

export interface Document {
  title: string
  source_type: DocumentSourceType
  chunk_count: number
  content_length: number
  timestamp: string
  url?: string
  source_path?: string
}

// =============================================================================
// RAG TYPES
// =============================================================================

export interface RAGChunk {
  content: string
  metadata: Record<string, unknown>
  relevance_score: number
}

export interface RAGQueryRequest {
  query: string
  workflow_id?: string
  doc_types?: string[]
  n_results?: number
}

export interface RAGQueryResponse {
  query: string
  results: RAGChunk[]
  total_found: number
}

export interface RAGStats {
  total_chunks: number
  total_documents: number
  doc_types: Record<string, number>
}

// =============================================================================
// WEBSOCKET TYPES
// =============================================================================

export interface WSMessage {
  type: 'connected' | 'progress' | 'stage_start' | 'stage_end' | 'error' | 'done' | 'pong'
  task_id?: string
  event?: string
  stage?: string
  message?: string
  timestamp?: string
  authenticated?: boolean
  username?: string
  workflow_id?: string
  data?: Record<string, unknown>
}

export interface WSProgressEvent {
  type: 'progress'
  task_id: string
  event: string
  stage: string
  message: string
  timestamp: string
  data?: Record<string, unknown>
}

// =============================================================================
// API RESPONSE TYPES
// =============================================================================

export interface HealthResponse {
  status: 'healthy' | 'degraded'
  ollama_reachable: boolean
  rag_enabled: boolean
  version: string
}

export interface ErrorResponse {
  detail: string
  error_code?: string
}

export interface Workflow {
  workflow_id: string
  stage_count: number
}