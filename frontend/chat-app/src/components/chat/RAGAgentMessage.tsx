import { memo, useState, useMemo } from 'react'
import { Database, ChevronDown, ChevronUp, Search, Check, AlertCircle } from 'lucide-react'
import clsx from 'clsx'

export interface RAGQuery {
  query: string
  iteration: number
  chunks_found: number
  relevance?: number
}

export interface RAGSearchGroup {
  event_id: string
  status: 'searching' | 'complete' | 'failed'
  total_iterations: number
  total_chunks: number
  final_relevance: number
  accepted: boolean
  queries: RAGQuery[]
  timestamp: string
}

interface RAGAgentMessageProps {
  group: RAGSearchGroup
  defaultExpanded?: boolean
}

function RelevanceIndicator({ score }: { score: number }) {
  const percentage = Math.round(score * 100)
  const color = score >= 0.7 ? 'text-green-600' : score >= 0.5 ? 'text-yellow-600' : 'text-red-500'
  
  return (
    <span className={clsx('text-xs font-medium', color)}>
      {percentage}%
    </span>
  )
}

function QueryItem({ query, isLast }: { query: RAGQuery; isLast: boolean }) {
  return (
    <div className={clsx(
      'relative pl-6 pb-3',
      !isLast && 'border-l-2 border-gray-200'
    )}>
      {/* Timeline dot */}
      <div className="absolute left-0 top-0 -translate-x-1/2 w-2.5 h-2.5 rounded-full bg-primary-400 border-2 border-white" />
      
      <div className="bg-gray-50 rounded-lg p-3 ml-2">
        <div className="flex items-center gap-2 text-xs text-gray-500 mb-1">
          <Search size={12} />
          <span>Iteration {query.iteration}</span>
          {query.relevance !== undefined && (
            <>
              <span className="text-gray-300">•</span>
              <RelevanceIndicator score={query.relevance} />
            </>
          )}
        </div>
        <p className="text-sm text-gray-700 font-mono break-words">
          {query.query}
        </p>
        <div className="text-xs text-gray-500 mt-1">
          Found {query.chunks_found} chunk{query.chunks_found !== 1 ? 's' : ''}
        </div>
      </div>
    </div>
  )
}

export const RAGAgentMessage = memo(function RAGAgentMessage({
  group,
  defaultExpanded = false,
}: RAGAgentMessageProps) {
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)
  
  const statusConfig = useMemo(() => {
    if (group.status === 'searching') {
      return {
        icon: <Database size={16} className="animate-pulse" />,
        text: 'Searching knowledge base...',
        bgColor: 'bg-blue-50 border-blue-200',
        textColor: 'text-blue-700',
      }
    }
    if (group.accepted) {
      return {
        icon: <Check size={16} />,
        text: `Found ${group.total_chunks} relevant chunks`,
        bgColor: 'bg-green-50 border-green-200',
        textColor: 'text-green-700',
      }
    }
    return {
      icon: <AlertCircle size={16} />,
      text: 'Low relevance results',
      bgColor: 'bg-yellow-50 border-yellow-200',
      textColor: 'text-yellow-700',
    }
  }, [group.status, group.accepted, group.total_chunks])

  const isSearching = group.status === 'searching'

  return (
    <div className="flex gap-4 p-4 bg-white">
      {/* Avatar */}
      <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 bg-purple-100 text-purple-600">
        <Database size={18} />
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-2">
          <span className="font-medium text-gray-900">RAG Agent</span>
          {isSearching && (
            <span className="text-xs text-primary-600 animate-pulse">
              searching...
            </span>
          )}
        </div>

        {/* Collapsible Card */}
        <div className={clsx(
          'border rounded-lg overflow-hidden transition-colors',
          statusConfig.bgColor
        )}>
          {/* Header - always visible */}
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            disabled={isSearching}
            className={clsx(
              'w-full flex items-center justify-between p-3 text-left',
              'hover:bg-white/50 transition-colors',
              'disabled:cursor-default disabled:hover:bg-transparent'
            )}
          >
            <div className="flex items-center gap-2">
              <span className={statusConfig.textColor}>
                {statusConfig.icon}
              </span>
              <span className={clsx('font-medium text-sm', statusConfig.textColor)}>
                {statusConfig.text}
              </span>
            </div>
            
            <div className="flex items-center gap-3">
              {!isSearching && (
                <>
                  <div className="text-xs text-gray-500">
                    {group.total_iterations} queries
                  </div>
                  <RelevanceIndicator score={group.final_relevance} />
                  <span className="text-gray-400">
                    {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                  </span>
                </>
              )}
            </div>
          </button>

          {/* Expanded content */}
          {isExpanded && group.queries.length > 0 && (
            <div className="border-t border-gray-200 bg-white p-4">
              <div className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-3">
                Search Iterations
              </div>
              <div className="space-y-0">
                {group.queries.map((query, idx) => (
                  <QueryItem
                    key={`${query.iteration}-${idx}`}
                    query={query}
                    isLast={idx === group.queries.length - 1}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
})

export default RAGAgentMessage