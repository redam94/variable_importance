import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import type { ChatMessage, RAGSearchGroup, WSMessage } from '../types/ws'

// =============================================================================
// TYPES
// =============================================================================

export interface ChatSession {
  sessionId: string
  workflowId: string
  createdAt: string
  updatedAt: string
  messages: ChatMessage[]
  wsMessages: WSMessage[]
  isRunning: boolean
  currentTaskId: string | null
  currentStage: string | null
  progress: number
  startedAt: string | null
  ragGroups: Record<string, RAGSearchGroup>
}

interface ChatState {
  // Sessions keyed by pageType (e.g., 'workflow-chat', 'rag-chat')
  sessions: Record<string, ChatSession>
}

interface ChatActions {
  // Session management
  initSession: (pageType: string, workflowId?: string) => void
  loadWorkflow: (pageType: string, workflowId: string) => void
  newSession: (pageType: string) => void

  // Message actions
  addMessage: (pageType: string, message: ChatMessage) => void
  addWsMessage: (pageType: string, message: WSMessage) => void
  clearMessages: (pageType: string) => void

  // Workflow state actions
  startWorkflowRun: (pageType: string, taskId: string) => void
  updateWorkflowProgress: (pageType: string, updates: { currentStage?: string | null; progress?: number }) => void
  completeWorkflowRun: (pageType: string) => void
  failWorkflowRun: (pageType: string) => void

  // RAG tracking
  updateRagGroup: (pageType: string, eventId: string, group: RAGSearchGroup) => void

  // Backward-compatible methods for Workflows page
  setWorkflowId: (workflowId: string) => void
  clearAllMessages: () => void
}

type ChatStore = ChatState & ChatActions

// =============================================================================
// HELPERS
// =============================================================================

function generateWorkflowId(): string {
  const now = new Date()
  return `workflow_${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}`
}

function generateSessionId(): string {
  return `session_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
}

function createEmptySession(workflowId?: string): ChatSession {
  const now = new Date().toISOString()
  return {
    sessionId: generateSessionId(),
    workflowId: workflowId || generateWorkflowId(),
    createdAt: now,
    updatedAt: now,
    messages: [],
    wsMessages: [],
    isRunning: false,
    currentTaskId: null,
    currentStage: null,
    progress: 0,
    startedAt: null,
    ragGroups: {},
  }
}

// =============================================================================
// STORE
// =============================================================================

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      sessions: {},

      initSession: (pageType: string, workflowId?: string) => {
        const state = get()
        const sessions = state.sessions ?? {}
        // Only create if doesn't exist
        if (!sessions[pageType]) {
          set({
            sessions: {
              ...sessions,
              [pageType]: createEmptySession(workflowId),
            },
          })
        }
      },

      loadWorkflow: (pageType: string, workflowId: string) => {
        set((state) => ({
          sessions: {
            ...state.sessions,
            [pageType]: createEmptySession(workflowId),
          },
        }))
      },

      newSession: (pageType: string) => {
        set((state) => ({
          sessions: {
            ...state.sessions,
            [pageType]: createEmptySession(),
          },
        }))
      },

      addMessage: (pageType: string, message: ChatMessage) => {
        set((state) => {
          const session = state.sessions[pageType]
          if (!session) return state

          return {
            sessions: {
              ...state.sessions,
              [pageType]: {
                ...session,
                messages: [...session.messages, message],
                updatedAt: new Date().toISOString(),
              },
            },
          }
        })
      },

      addWsMessage: (pageType: string, message: WSMessage) => {
        set((state) => {
          const session = state.sessions[pageType]
          if (!session) return state

          return {
            sessions: {
              ...state.sessions,
              [pageType]: {
                ...session,
                wsMessages: [...session.wsMessages.slice(-49), message],
                updatedAt: new Date().toISOString(),
              },
            },
          }
        })
      },

      clearMessages: (pageType: string) => {
        set((state) => {
          const session = state.sessions[pageType]
          if (!session) return state

          return {
            sessions: {
              ...state.sessions,
              [pageType]: {
                ...session,
                messages: [],
                wsMessages: [],
                ragGroups: {},
                isRunning: false,
                currentTaskId: null,
                currentStage: null,
                progress: 0,
                startedAt: null,
                updatedAt: new Date().toISOString(),
              },
            },
          }
        })
      },

      startWorkflowRun: (pageType: string, taskId: string) => {
        set((state) => {
          const session = state.sessions[pageType]
          if (!session) return state

          return {
            sessions: {
              ...state.sessions,
              [pageType]: {
                ...session,
                isRunning: true,
                currentTaskId: taskId,
                currentStage: null,
                progress: 0,
                startedAt: new Date().toISOString(),
                wsMessages: [],
                ragGroups: {},
                updatedAt: new Date().toISOString(),
              },
            },
          }
        })
      },

      updateWorkflowProgress: (pageType: string, updates) => {
        set((state) => {
          const session = state.sessions[pageType]
          if (!session) return state

          return {
            sessions: {
              ...state.sessions,
              [pageType]: {
                ...session,
                ...updates,
                updatedAt: new Date().toISOString(),
              },
            },
          }
        })
      },

      completeWorkflowRun: (pageType: string) => {
        set((state) => {
          const session = state.sessions[pageType]
          if (!session) return state

          return {
            sessions: {
              ...state.sessions,
              [pageType]: {
                ...session,
                isRunning: false,
                progress: 100,
                currentStage: 'complete',
                updatedAt: new Date().toISOString(),
              },
            },
          }
        })
      },

      failWorkflowRun: (pageType: string) => {
        set((state) => {
          const session = state.sessions[pageType]
          if (!session) return state

          return {
            sessions: {
              ...state.sessions,
              [pageType]: {
                ...session,
                isRunning: false,
                currentStage: 'error',
                updatedAt: new Date().toISOString(),
              },
            },
          }
        })
      },

      updateRagGroup: (pageType: string, eventId: string, group: RAGSearchGroup) => {
        set((state) => {
          const session = state.sessions[pageType]
          if (!session) return state

          return {
            sessions: {
              ...state.sessions,
              [pageType]: {
                ...session,
                ragGroups: {
                  ...session.ragGroups,
                  [eventId]: group,
                },
                updatedAt: new Date().toISOString(),
              },
            },
          }
        })
      },

      // Backward-compatible: set workflow for workflow-chat page
      setWorkflowId: (workflowId: string) => {
        set((state) => ({
          sessions: {
            ...state.sessions,
            'workflow-chat': createEmptySession(workflowId),
          },
        }))
      },

      // Backward-compatible: clear messages for workflow-chat page
      clearAllMessages: () => {
        set((state) => {
          const session = state.sessions['workflow-chat']
          if (!session) return state

          return {
            sessions: {
              ...state.sessions,
              'workflow-chat': {
                ...session,
                messages: [],
                wsMessages: [],
                ragGroups: {},
                isRunning: false,
                currentTaskId: null,
                currentStage: null,
                progress: 0,
                startedAt: null,
                updatedAt: new Date().toISOString(),
              },
            },
          }
        })
      },
    }),
    {
      name: 'chat-storage-v2',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        sessions: state.sessions,
      }),
    }
  )
)

// =============================================================================
// SELECTORS - Use these to avoid re-renders
// =============================================================================

export const selectSession = (pageType: string) => (state: ChatStore) => 
  state.sessions?.[pageType]

export const selectMessages = (pageType: string) => (state: ChatStore) => 
  state.sessions?.[pageType]?.messages ?? []

export const selectWsMessages = (pageType: string) => (state: ChatStore) => 
  state.sessions?.[pageType]?.wsMessages ?? []

export const selectIsRunning = (pageType: string) => (state: ChatStore) => 
  state.sessions?.[pageType]?.isRunning ?? false

export const selectWorkflowId = (pageType: string) => (state: ChatStore) => 
  state.sessions?.[pageType]?.workflowId ?? ''

export const selectCurrentTaskId = (pageType: string) => (state: ChatStore) => 
  state.sessions?.[pageType]?.currentTaskId ?? null

export const selectCurrentStage = (pageType: string) => (state: ChatStore) => 
  state.sessions?.[pageType]?.currentStage ?? null

export const selectProgress = (pageType: string) => (state: ChatStore) => 
  state.sessions?.[pageType]?.progress ?? 0

export const selectStartedAt = (pageType: string) => (state: ChatStore) => 
  state.sessions?.[pageType]?.startedAt ?? null

export const selectRagGroups = (pageType: string) => (state: ChatStore) => 
  state.sessions?.[pageType]?.ragGroups ?? {}

// Convenience selector for Workflows page - gets workflow-chat workflowId
export const selectCurrentWorkflowId = (state: ChatStore) => 
  state.sessions?.['workflow-chat']?.workflowId ?? ''