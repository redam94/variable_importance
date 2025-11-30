import { useState, useRef, useEffect } from 'react'
import { Send, StopCircle } from 'lucide-react'
import clsx from 'clsx'

interface ChatInputProps {
  onSend: (message: string) => void
  onStop?: () => void
  isLoading?: boolean
  placeholder?: string
  disabled?: boolean
}

export function ChatInput({
  onSend,
  onStop,
  isLoading,
  placeholder = 'Type your message...',
  disabled,
}: ChatInputProps) {
  const [message, setMessage] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  
  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current
    if (textarea) {
      textarea.style.height = 'auto'
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`
    }
  }, [message])
  
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (message.trim() && !isLoading && !disabled) {
      onSend(message.trim())
      setMessage('')
    }
  }
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit(e)
    }
  }
  
  return (
    <form onSubmit={handleSubmit} className="border-t border-gray-200 bg-white p-4">
      <div className="flex items-end gap-3 max-w-4xl mx-auto">
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled || isLoading}
            rows={1}
            className={clsx(
              'w-full px-4 py-3 pr-12 rounded-xl border border-gray-300 resize-none',
              'focus:ring-2 focus:ring-primary-500 focus:border-transparent',
              'outline-none transition-shadow',
              'disabled:bg-gray-50 disabled:text-gray-500'
            )}
          />
        </div>
        
        {isLoading ? (
          <button
            type="button"
            onClick={onStop}
            className="p-3 rounded-xl bg-red-500 text-white hover:bg-red-600 transition-colors"
            title="Stop generation"
          >
            <StopCircle size={22} />
          </button>
        ) : (
          <button
            type="submit"
            disabled={!message.trim() || disabled}
            className={clsx(
              'p-3 rounded-xl transition-colors',
              message.trim() && !disabled
                ? 'gradient-bg text-white hover:opacity-90'
                : 'bg-gray-100 text-gray-400'
            )}
            title="Send message"
          >
            <Send size={22} />
          </button>
        )}
      </div>
      
      <p className="text-xs text-gray-400 text-center mt-2">
        Press <kbd className="px-1 py-0.5 bg-gray-100 rounded">Enter</kbd> to send, 
        <kbd className="px-1 py-0.5 bg-gray-100 rounded ml-1">Shift+Enter</kbd> for new line
      </p>
    </form>
  )
}