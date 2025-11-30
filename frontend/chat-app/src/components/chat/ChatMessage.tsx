import { memo } from 'react'
import { User, Bot } from 'lucide-react'
import clsx from 'clsx'

interface ChatMessageProps {
  role: 'user' | 'assistant'
  content: string
  isStreaming?: boolean
}

export const ChatMessage = memo(function ChatMessage({
  role,
  content,
  isStreaming,
}: ChatMessageProps) {
  const isUser = role === 'user'
  
  return (
    <div
      className={clsx(
        'flex gap-4 p-4',
        isUser ? 'bg-gray-50' : 'bg-white'
      )}
    >
      {/* Avatar */}
      <div
        className={clsx(
          'w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0',
          isUser ? 'bg-primary-100 text-primary-600' : 'gradient-bg text-white'
        )}
      >
        {isUser ? <User size={18} /> : <Bot size={18} />}
      </div>
      
      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-medium text-gray-900">
            {isUser ? 'You' : 'Assistant'}
          </span>
          {isStreaming && (
            <span className="text-xs text-primary-600 animate-pulse">
              typing...
            </span>
          )}
        </div>
        
        <div className="prose-chat text-gray-700 whitespace-pre-wrap">
          {content || (
            <span className="loading-dots">
              <span className="inline-block w-2 h-2 bg-gray-400 rounded-full mx-0.5">•</span>
              <span className="inline-block w-2 h-2 bg-gray-400 rounded-full mx-0.5">•</span>
              <span className="inline-block w-2 h-2 bg-gray-400 rounded-full mx-0.5">•</span>
            </span>
          )}
        </div>
      </div>
    </div>
  )
})