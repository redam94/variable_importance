import { Outlet, NavLink, useNavigate } from 'react-router-dom'
import { useAuthStore } from '../../stores/authStore'
import {
  LayoutDashboard,
  MessageSquare,
  FileText,
  FolderOpen,
  Search,
  LogOut,
  User,
  Bot,
  Menu,
  X,
} from 'lucide-react'
import { useState } from 'react'
import clsx from 'clsx'

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/chat', icon: MessageSquare, label: 'Chat' },
  { to: '/agent', icon: Bot, label: 'RAG Agent' },
  { to: '/documents', icon: FileText, label: 'Documents' },
  { to: '/workflows', icon: FolderOpen, label: 'Workflows' },
  { to: '/rag', icon: Search, label: 'RAG Search' },
]

export function Layout() {
  const { user, logout } = useAuthStore()
  const navigate = useNavigate()
  const [sidebarOpen, setSidebarOpen] = useState(true)
  
  const handleLogout = () => {
    logout()
    navigate('/login')
  }
  
  return (
    <div className="flex h-full bg-gray-50">
      {/* Sidebar */}
      <aside
        className={clsx(
          'fixed inset-y-0 left-0 z-50 flex flex-col bg-white border-r border-gray-200 transition-all duration-300',
          sidebarOpen ? 'w-64' : 'w-16'
        )}
      >
        {/* Logo */}
        <div className="flex items-center justify-between h-16 px-4 border-b border-gray-200">
          {sidebarOpen && (
            <h1 className="text-xl font-bold gradient-text">🤖 DS Agent</h1>
          )}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
          >
            {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
        
        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                clsx(
                  'sidebar-link',
                  isActive && 'active',
                  !sidebarOpen && 'justify-center px-2'
                )
              }
            >
              <item.icon size={20} />
              {sidebarOpen && <span>{item.label}</span>}
            </NavLink>
          ))}
        </nav>
        
        {/* User section */}
        <div className="border-t border-gray-200 p-3">
          {sidebarOpen ? (
            <div className="flex items-center gap-3 px-2 py-2">
              <div className="w-8 h-8 rounded-full gradient-bg flex items-center justify-center text-white">
                <User size={16} />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 truncate">
                  {user?.username || 'User'}
                </p>
                <p className="text-xs text-gray-500 truncate">
                  {user?.email || 'Logged in'}
                </p>
              </div>
              <button
                onClick={handleLogout}
                className="p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
                title="Logout"
              >
                <LogOut size={18} />
              </button>
            </div>
          ) : (
            <button
              onClick={handleLogout}
              className="w-full p-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors flex justify-center"
              title="Logout"
            >
              <LogOut size={20} />
            </button>
          )}
        </div>
      </aside>
      
      {/* Main content */}
      <main
        className={clsx(
          'flex-1 flex flex-col overflow-hidden transition-all duration-300',
          sidebarOpen ? 'ml-64' : 'ml-16'
        )}
      >
        <Outlet />
      </main>
    </div>
  )
}