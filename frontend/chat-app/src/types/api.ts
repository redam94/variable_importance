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
  workflow_id: string
  stage_name?: string
  model?: string
  data_path?: string
  web_search_enabled?: boolean
  rag_enabled?: boolean
}

export interface WorkflowMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp?: string
}

export interface WorkflowResponse {
  workflow_id: string
  stage_name: string
  status: string
  message: WorkflowMessage
  action_taken?: string
  code_executed?: string
  plots?: string[]
  summary?: string
  error?: string
  artifacts: string[]
  execution_time_seconds: number
}

export interface DataFileUploadResponse {
  success: boolean
  filename: string
  file_path: string
  file_size: number
  content_type: string
  message: string
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

export interface Workflow {
  workflow_id: string
  created_at: string
  stage_count: number
  stages: string[]
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
  title: string
  url: string
  chunk_count: number
  content_length: number
  workflow_id: string
  message: string
}

export interface Document {
  title: string
  source_type: DocumentSourceType
  chunk_count: number
  created_at: string
}

// =============================================================================
// RAG TYPES
// =============================================================================

export interface RAGQueryRequest {
  query: string
  workflow_id: string
  n_results?: number
}

export interface RAGChunk {
  content: string
  source: string
  score: number
  metadata: Record<string, unknown>
}

export interface RAGQueryResponse {
  query: string
  results: RAGChunk[]
  total_chunks: number
}

export interface RAGStats {
  workflow_id: string
  total_documents: number
  total_chunks: number
  sources: string[]
}

// =============================================================================
// HEALTH TYPES
// =============================================================================

export interface HealthResponse {
  status: string
  version: string
  ollama_connected: boolean
  rag_available: boolean
  workflow_ready: boolean
}

// =============================================================================
// IMAGE TYPES
// =============================================================================

export interface ImageInfo {
  filename: string
  path: string       // Relative path from workflow root
  stage: string
  url: string        // API URL to fetch the image (includes query param)
  size_bytes: number
}

export interface StageFiles {
  stage_name: string
  images: ImageInfo[]
  data_files: string[]
  code_files: string[]
}

export interface WorkflowImages {
  workflow_id: string
  total_images: number
  images: ImageInfo[]
}