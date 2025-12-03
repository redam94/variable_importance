import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

// =============================================================================
// TYPES
// =============================================================================

interface ChatState {
  // Workflow identity (persisted)
  workflowId: string

  // Actions
  setWorkflowId: (id: string) => void
  generateNewWorkflowId: () => string
}

// =============================================================================
// HELPERS
// =============================================================================

function generateWorkflowId(): string {
  const now = new Date()
  return `workflow_${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}_${String(now.getHours()).padStart(2, '0')}${String(now.getMinutes()).padStart(2, '0')}${String(now.getSeconds()).padStart(2, '0')}`
}

// =============================================================================
// STORE
// =============================================================================

export const useChatStore = create<ChatState>()(
  persist(
    (set) => ({
      workflowId: generateWorkflowId(),

      setWorkflowId: (id: string) => set({ workflowId: id }),

      generateNewWorkflowId: () => {
        const newId = generateWorkflowId()
        set({ workflowId: newId })
        return newId
      },
    }),
    {
      name: 'chat-storage',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        workflowId: state.workflowId,
      }),
    }
  )
)