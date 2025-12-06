import { memo, useMemo } from 'react'
import { User, Bot, Copy, Check } from 'lucide-react'
import clsx from 'clsx'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { useState } from 'react'

// Import KaTeX CSS - add this to your index.html or import in main.tsx:
// <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" />

interface ChatMessageProps {
  role: 'user' | 'assistant'
  content: string
  isStreaming?: boolean
}

// Copy button for code blocks
function CopyButton({ code }: { code: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <button
      onClick={handleCopy}
      className="absolute top-2 right-2 p-1.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-300 hover:text-white transition-colors"
      aria-label="Copy code"
    >
      {copied ? <Check size={14} /> : <Copy size={14} />}
    </button>
  )
}

// Markdown component overrides
const markdownComponents = {
  // Code blocks and inline code
  code({ node, inline, className, children, ...props }: any) {
    const match = /language-(\w+)/.exec(className || '')
    const codeString = String(children).replace(/\n$/, '')

    if (!inline && match) {
      return (
        <div className="relative group my-3">
          <div className="absolute top-2 left-3 text-xs text-gray-400 font-mono">
            {match[1]}
          </div>
          <CopyButton code={codeString} />
          <SyntaxHighlighter
            style={oneDark}
            language={match[1]}
            PreTag="div"
            className="rounded-lg !mt-0 !pt-8"
            {...props}
          >
            {codeString}
          </SyntaxHighlighter>
        </div>
      )
    }

    // Inline code (but not math)
    return (
      <code
        className="px-1.5 py-0.5 bg-gray-100 text-gray-800 rounded text-sm font-mono"
        {...props}
      >
        {children}
      </code>
    )
  },

  // Paragraphs
  p({ children }: any) {
    return <p className="mb-3 last:mb-0 leading-relaxed">{children}</p>
  },

  // Headers
  h1: ({ children }: any) => <h1 className="text-xl font-bold mb-3 mt-4 first:mt-0">{children}</h1>,
  h2: ({ children }: any) => <h2 className="text-lg font-bold mb-2 mt-4 first:mt-0">{children}</h2>,
  h3: ({ children }: any) => <h3 className="text-base font-bold mb-2 mt-3 first:mt-0">{children}</h3>,

  // Lists
  ul: ({ children }: any) => <ul className="list-disc list-inside mb-3 space-y-1">{children}</ul>,
  ol: ({ children }: any) => <ol className="list-decimal list-inside mb-3 space-y-1">{children}</ol>,
  li: ({ children }: any) => <li className="leading-relaxed">{children}</li>,

  // Links
  a: ({ href, children }: any) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-primary-600 hover:text-primary-700 underline"
    >
      {children}
    </a>
  ),

  // Blockquotes
  blockquote: ({ children }: any) => (
    <blockquote className="border-l-4 border-gray-300 pl-4 my-3 italic text-gray-600">
      {children}
    </blockquote>
  ),

  // Tables
  table: ({ children }: any) => (
    <div className="overflow-x-auto my-3">
      <table className="min-w-full border-collapse border border-gray-300">{children}</table>
    </div>
  ),
  thead: ({ children }: any) => <thead className="bg-gray-100">{children}</thead>,
  th: ({ children }: any) => (
    <th className="border border-gray-300 px-3 py-2 text-left font-semibold">{children}</th>
  ),
  td: ({ children }: any) => (
    <td className="border border-gray-300 px-3 py-2">{children}</td>
  ),

  // Horizontal rule
  hr: () => <hr className="my-4 border-gray-300" />,

  // Strong/Bold
  strong: ({ children }: any) => <strong className="font-semibold">{children}</strong>,

  // Emphasis/Italic
  em: ({ children }: any) => <em className="italic">{children}</em>,
}

export const ChatMessage = memo(function ChatMessage({
  role,
  content,
  isStreaming,
}: ChatMessageProps) {
  const isUser = role === 'user'

  // Memoize markdown parsing for performance
  const renderedContent = useMemo(() => {
    if (!content) return null
    return (
      <ReactMarkdown 
        remarkPlugins={[remarkGfm, remarkMath]} 
        rehypePlugins={[rehypeKatex]}
        components={markdownComponents}
      >
        {content}
      </ReactMarkdown>
    )
  }, [content])

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

        <div className="text-gray-700 prose prose-sm max-w-none">
          {renderedContent || (
            <span className="flex gap-1">
              <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </span>
          )}
        </div>
      </div>
    </div>
  )
})

export default ChatMessage