import { useState } from 'react'
import { Code, Image, FileText, ChevronDown, ChevronUp, Copy, Check } from 'lucide-react'
import clsx from 'clsx'

interface WorkflowOutputProps {
  code?: string | null
  plots?: string[]
  summary?: string | null
  error?: string | null
  actionTaken?: string | null
}

export function WorkflowOutput({ code, plots, summary, error, actionTaken }: WorkflowOutputProps) {
  const [showCode, setShowCode] = useState(false)
  const [copied, setCopied] = useState(false)

  const hasOutput = code || (plots && plots.length > 0) || summary || error

  if (!hasOutput) return null

  const handleCopyCode = async () => {
    if (code) {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  return (
    <div className="space-y-4">
      {/* Action badge */}
      {actionTaken && (
        <div className="flex items-center gap-2">
          <span className={clsx(
            'px-2 py-1 rounded-full text-xs font-medium',
            actionTaken === 'execute' ? 'bg-purple-100 text-purple-700' : 'bg-blue-100 text-blue-700'
          )}>
            {actionTaken === 'execute' ? '🔧 Code Executed' : '💡 Answered from Context'}
          </span>
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Code section */}
      {code && (
        <div className="bg-gray-900 rounded-lg overflow-hidden">
          <button
            onClick={() => setShowCode(!showCode)}
            className="w-full flex items-center justify-between px-4 py-2 bg-gray-800 text-gray-300 hover:bg-gray-700 transition-colors"
          >
            <span className="flex items-center gap-2 text-sm font-medium">
              <Code size={16} />
              Generated Code
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  handleCopyCode()
                }}
                className="p-1 hover:bg-gray-600 rounded"
                title="Copy code"
              >
                {copied ? <Check size={14} className="text-green-400" /> : <Copy size={14} />}
              </button>
              {showCode ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </div>
          </button>
          {showCode && (
            <pre className="p-4 overflow-x-auto text-sm text-gray-300 max-h-96">
              <code>{code}</code>
            </pre>
          )}
        </div>
      )}

      {/* Plots section */}
      {plots && plots.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-sm font-medium text-gray-700">
            <Image size={16} />
            Generated Plots ({plots.length})
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {plots.map((plot, index) => (
              <div key={index} className="bg-white border rounded-lg overflow-hidden shadow-sm">
                <img
                  src={plot.startsWith('data:') ? plot : `data:image/png;base64,${plot}`}
                  alt={`Plot ${index + 1}`}
                  className="w-full h-auto"
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Summary section */}
      {summary && (
        <div className="bg-white border rounded-lg p-4">
          <div className="flex items-center gap-2 text-sm font-medium text-gray-700 mb-2">
            <FileText size={16} />
            Summary
          </div>
          <div className="prose prose-sm max-w-none text-gray-600 whitespace-pre-wrap">
            {summary}
          </div>
        </div>
      )}
    </div>
  )
}