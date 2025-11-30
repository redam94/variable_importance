import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { ChatMessage } from '../types/api'

interface ChatState {
  workflowId: string
  messages: ChatMessage[]
  isStreaming: boolean
  streamingContent: string
  error: string | null
  
  setWorkflowId: (id: string) => void
  addMessage: (message: ChatMessage) => void
  updateLastMessage: (content: string) => void
  appendToStream: (content: string) => void
  startStreaming: () => void
  stopStreaming: () => void
  clearMessages: () => void
  setError: (error: string | null) => void
}

function generateWorkflowId(): string {
  const now = new Date()
  return `workflow_${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}`
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      workflowId: generateWorkflowId(),
      messages: [],
      isStreaming: false,
      streamingContent: '',
      error: null,
      
      setWorkflowId: (id: string) => set({ workflowId: id }),
      
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
            messages: [...state.messages, { role: 'assistant', content: streamingContent }],
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
          workflowId: generateWorkflowId(),
        }),
      
      setError: (error: string | null) => set({ error }),
    }),
    {
      name: 'chat-storage',
      partialize: (state) => ({
        workflowId: state.workflowId,
        messages: state.messages,
      }),
    }
  )
)