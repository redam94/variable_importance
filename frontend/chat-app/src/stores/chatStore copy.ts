import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import type { ChatMessage, WSMessage } from '../types/api'

// =============================================================================
// TYPES
// =============================================================================

export interface WorkflowRunState {
  isRunning: boolean
  currentTaskId: string | null
  currentStage: string | null
  progress: number
  startedAt: string | null
  wsMessages: WSMessage[] // Recent progress messages for display
}

interface ChatState {
  // Workflow identity
  workflowId: string

  // Chat messages (persisted)
  messages: ChatMessage[]

  // Streaming state (not persisted)
  isStreaming: boolean
  streamingContent: string
  error: string | null

  // Workflow run state (persisted for cross-page tracking)
  workflowRun: WorkflowRunState

  // Actions
  setWorkflowId: (id: string) => void
  addMessage: (message: ChatMessage) => void
  updateLastMessage: (content: string) => void
  appendToStream: (content: string) => void
  startStreaming: () => void
  stopStreaming: () => void
  clearMessages: () => void
  setError: (error: string | null) => void

  // Workflow run actions
  startWorkflowRun: (taskId: string) => void
  updateWorkflowProgress: (stage: string | null, progress: number) => void
  addWsMessage: (message: WSMessage) => void
  completeWorkflowRun: () => void
  failWorkflowRun: (error?: string) => void
  clearWorkflowRun: () => void
}

// =============================================================================
// HELPERS
// =============================================================================

function generateWorkflowId(): string {
  const now = new Date()
  return `workflow_${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}`
}

const INITIAL_WORKFLOW_RUN: WorkflowRunState = {
  isRunning: false,
  currentTaskId: null,
  currentStage: null,
  progress: 0,
  startedAt: null,
  wsMessages: [],
}

// =============================================================================
// STORE
// =============================================================================

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      // Initial state
      workflowId: generateWorkflowId(),
      messages: [],
      isStreaming: false,
      streamingContent: '',
      error: null,
      workflowRun: { ...INITIAL_WORKFLOW_RUN },

      // Workflow identity
      setWorkflowId: (id: string) =>
        set(() => ({
          workflowId: id,
        })),

      // Message management
      addMessage: (message: ChatMessage) =>
        set((state) => ({
          messages: [...state.messages, message],
          streamingContent: '',
        })),

      updateLastMessage: (content: string) =>
        set((state) => {
          const messages = [...state.messages]
          if (messages.length > 0) {
            messages[messages.length - 1] = {
              ...messages[messages.length - 1],
              content,
            }
          }
          return { messages }
        }),

      appendToStream: (content: string) =>
        set((state) => ({
          streamingContent: state.streamingContent + content,
        })),

      startStreaming: () =>
        set({
          isStreaming: true,
          streamingContent: '',
          error: null,
        }),

      stopStreaming: () => {
        const { streamingContent } = get()
        if (streamingContent) {
          set((state) => ({
            messages: [
              ...state.messages,
              { role: 'assistant', content: streamingContent },
            ],
            isStreaming: false,
            streamingContent: '',
          }))
        } else {
          set({ isStreaming: false })
        }
      },

      clearMessages: () =>
        set({
          messages: [],
          streamingContent: '',
          workflowRun: { ...INITIAL_WORKFLOW_RUN },
        }),

      setError: (error: string | null) => set({ error }),

      // =============================================================================
      // WORKFLOW RUN STATE MANAGEMENT
      // =============================================================================

      startWorkflowRun: (taskId: string) =>
        set(() => ({
          workflowRun: {
            isRunning: true,
            currentTaskId: taskId,
            currentStage: null,
            progress: 0,
            startedAt: new Date().toISOString(),
            wsMessages: [],
          },
          error: null,
        })),

      updateWorkflowProgress: (stage: string | null, progress: number) =>
        set((state) => ({
          workflowRun: {
            ...state.workflowRun,
            currentStage: stage ?? state.workflowRun.currentStage,
            progress,
          },
        })),

      addWsMessage: (message: WSMessage) =>
        set((state) => ({
          workflowRun: {
            ...state.workflowRun,
            // Keep last 50 messages
            wsMessages: [...state.workflowRun.wsMessages.slice(-49), message],
            // Update stage and progress from message
            currentStage: message.stage ?? state.workflowRun.currentStage,
            progress:
              message.type === 'done'
                ? 100
                : message.data?.progress !== undefined
                  ? Number(message.data.progress)
                  : state.workflowRun.progress,
          },
        })),

      completeWorkflowRun: () =>
        set((state) => ({
          workflowRun: {
            ...state.workflowRun,
            isRunning: false,
            progress: 100,
            currentStage: 'complete',
          },
        })),

      failWorkflowRun: (error?: string) =>
        set((state) => ({
          workflowRun: {
            ...state.workflowRun,
            isRunning: false,
          },
          error: error ?? 'Workflow failed',
        })),

      clearWorkflowRun: () =>
        set(() => ({
          workflowRun: { ...INITIAL_WORKFLOW_RUN },
          error: null,
        })),
    }),
    {
      name: 'chat-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        workflowId: state.workflowId,
        messages: state.messages,
        workflowRun: state.workflowRun,
      }),
    }
  )
)