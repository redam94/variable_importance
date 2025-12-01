import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { Layout } from './components/layout/Layout'
import { ProtectedRoute } from './components/layout/ProtectedRoute'
import { LoginPage } from './pages/Login'
import { RegisterPage } from './pages/Register'
import { DashboardPage } from './pages/Dashboard'
import { WorkflowChatPage } from './pages/Chat'
import { AgentChatPage } from './pages/AgentChat'
import { DocumentsPage } from './pages/Documents'
import { WorkflowsPage } from './pages/Workflows'
import { RAGSearchPage } from './pages/RAGSearch'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Public routes */}
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        
        {/* Protected routes with layout */}
        <Route
          element={
            <ProtectedRoute>
              <Layout />
            </ProtectedRoute>
          }
        >
          <Route path="/" element={<DashboardPage />} />
          <Route path="/chat" element={<WorkflowChatPage />} />
          <Route path="/agent" element={<AgentChatPage />} />
          <Route path="/documents" element={<DocumentsPage />} />
          <Route path="/workflows" element={<WorkflowsPage />} />
          <Route path="/rag" element={<RAGSearchPage />} />
        </Route>
        
        {/* Catch all */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}