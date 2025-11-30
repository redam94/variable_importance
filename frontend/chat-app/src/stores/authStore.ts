import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { User } from '../types/api'
import { authApi, getStoredToken, clearTokens } from '../lib/api'

interface AuthState {
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null
  
  login: (username: string, password: string) => Promise<void>
  logout: () => void
  checkAuth: () => Promise<void>
  clearError: () => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      isAuthenticated: false,
      isLoading: true,
      error: null,
      
      login: async (username: string, password: string) => {
        set({ isLoading: true, error: null })
        try {
          await authApi.login({ username, password })
          const user = await authApi.getCurrentUser()
          set({ user, isAuthenticated: true, isLoading: false })
        } catch (error) {
          const message = error instanceof Error ? error.message : 'Login failed'
          set({ error: message, isLoading: false })
          throw error
        }
      },
      
      logout: () => {
        authApi.logout()
        set({ user: null, isAuthenticated: false, error: null })
      },
      
      checkAuth: async () => {
        const token = getStoredToken()
        if (!token) {
          set({ isLoading: false, isAuthenticated: false })
          return
        }
        
        try {
          const user = await authApi.getCurrentUser()
          set({ user, isAuthenticated: true, isLoading: false })
        } catch {
          clearTokens()
          set({ user: null, isAuthenticated: false, isLoading: false })
        }
      },
      
      clearError: () => set({ error: null }),
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        user: state.user,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
)