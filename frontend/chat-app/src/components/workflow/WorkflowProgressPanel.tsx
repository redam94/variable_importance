import { useState, useEffect, useMemo } from 'react'
import clsx from 'clsx'
import {
  ChevronDown,
  ChevronRight,
  Clock,
  CheckCircle2,
  Loader2,
  AlertCircle,
  Search,
  Code,
  Brain,
  FileText,
  Sparkles,
  Database,
} from 'lucide-react'
import type { WSMessage } from '../../types/ws'

// =============================================================================
// TYPES
// =============================================================================

interface WorkflowProgressPanelProps {
  stage: string | null
  progress: number
  isRunning: boolean
  wsMessages: WSMessage[]
  startedAt: string | null
  className?: string
}

interface StageInfo {
  key: string
  label: string
  icon: React.ReactNode
  description: string
}

// =============================================================================
// CONSTANTS
// =============================================================================

const STAGES: StageInfo[] = [
  {
    key: 'gather_context',
    label: 'Context',
    icon: <Database size={14} />,
    description: 'Searching knowledge base & gathering relevant context',
  },
  {
    key: 'plan_and_decide',
    label: 'Plan',
    icon: <Brain size={14} />,
    description: 'Analyzing query and planning approach',
  },
  {
    key: 'execute',
    label: 'Execute',
    icon: <Code size={14} />,
    description: 'Running analysis code',
  },
  {
    key: 'answer',
    label: 'Answer',
    icon: <FileText size={14} />,
    description: 'Generating direct response',
  },
  {
    key: 'summarize',
    label: 'Summary',
    icon: <Sparkles size={14} />,
    description: 'Creating final summary',
  },
]

const MESSAGE_ICONS: Record<string, React.ReactNode> = {
  rag_query: <Search size={12} className="text-blue-500" />,
  rag_search_start: <Search size={12} className="text-blue-500" />,
  rag_search_end: <CheckCircle2 size={12} className="text-green-500" />,
  code_generated: <Code size={12} className="text-purple-500" />,
  execution_start: <Loader2 size={12} className="text-yellow-500 animate-spin" />,
  execution_result: <CheckCircle2 size={12} className="text-green-500" />,
  stage_start: <Loader2 size={12} className="text-blue-500 animate-spin" />,
  stage_end: <CheckCircle2 size={12} className="text-green-500" />,
  error: <AlertCircle size={12} className="text-red-500" />,
  progress: <Loader2 size={12} className="text-gray-400 animate-spin" />,
}

// =============================================================================
// HELPERS
// =============================================================================

function formatElapsed(startedAt: string | null): string {
  if (!startedAt) return '0s'
  const elapsed = Math.floor((Date.now() - new Date(startedAt).getTime()) / 1000)
  if (elapsed < 60) return `${elapsed}s`
  const mins = Math.floor(elapsed / 60)
  const secs = elapsed % 60
  return `${mins}m ${secs}s`
}

function formatTimestamp(ts: string): string {
  try {
    return new Date(ts).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  } catch {
    return ''
  }
}

function getStageStatus(
  stageKey: string,
  currentStage: string | null,
  wsMessages: WSMessage[]
): 'pending' | 'running' | 'completed' | 'error' {
  // Check for explicit stage events
  const stageEvents = wsMessages.filter((m) => m.stage === stageKey)
  const hasEnd = stageEvents.some((m) => m.type === 'stage_end' || m.event === 'stage_end')
  const hasError = stageEvents.some((m) => m.type === 'error')

  if (hasError) return 'error'
  if (hasEnd) return 'completed'
  if (currentStage === stageKey) return 'running'

  // Check if we're past this stage
  const stageIndex = STAGES.findIndex((s) => s.key === stageKey)
  const currentIndex = STAGES.findIndex((s) => s.key === currentStage)
  if (currentIndex > stageIndex) return 'completed'

  return 'pending'
}

function getMessageText(msg: WSMessage): string {
  // Prefer the message field
  if (msg.message) return msg.message

  // Build from type and data
  switch (msg.type) {
    case 'rag_query':
      return `RAG query: "${msg.data?.query}" (${msg.data?.chunks_found || 0} chunks)`
    case 'rag_search_start':
      return 'Starting RAG search...'
    case 'rag_search_end':
      return `RAG complete: ${msg.data?.total_chunks || 0} chunks, relevance ${((msg.data?.final_relevance || 0) * 100).toFixed(0)}%`
    case 'code_generated':
      return `Generated ${msg.data?.line_count || 0} lines of code`
    case 'execution_start':
      return 'Executing code...'
    case 'execution_result':
      return msg.data?.success ? 'Execution successful' : 'Execution failed'
    case 'stage_start':
      return `Started: ${msg.stage}`
    case 'stage_end':
      return `Completed: ${msg.stage}`
    case 'error':
      return `Error: ${msg.data?.error || 'Unknown error'}`
    case 'progress':
      return msg.event || 'Processing...'
    default:
      return msg.type
  }
}

// =============================================================================
// SUBCOMPONENTS
// =============================================================================

function StageIndicator({
  stage,
  status,
  isLast,
}: {
  stage: StageInfo
  status: 'pending' | 'running' | 'completed' | 'error'
  isLast: boolean
}) {
  return (
    <div className="flex items-center gap-1">
      <div
        className={clsx(
          'flex items-center justify-center w-6 h-6 rounded-full border-2 transition-all',
          {
            'border-gray-300 bg-gray-50 text-gray-400': status === 'pending',
            'border-blue-500 bg-blue-50 text-blue-600 animate-pulse': status === 'running',
            'border-green-500 bg-green-50 text-green-600': status === 'completed',
            'border-red-500 bg-red-50 text-red-600': status === 'error',
          }
        )}
        title={stage.description}
      >
        {status === 'running' ? <Loader2 size={12} className="animate-spin" /> : stage.icon}
      </div>
      <span
        className={clsx('text-xs font-medium', {
          'text-gray-400': status === 'pending',
          'text-blue-600': status === 'running',
          'text-green-600': status === 'completed',
          'text-red-600': status === 'error',
        })}
      >
        {stage.label}
      </span>
      {!isLast && (
        <div
          className={clsx('w-4 h-0.5 mx-1', {
            'bg-gray-200': status === 'pending',
            'bg-blue-300': status === 'running',
            'bg-green-300': status === 'completed',
            'bg-red-300': status === 'error',
          })}
        />
      )}
    </div>
  )
}

function ActivityItem({ message, isRunning }: { message: WSMessage; isRunning: boolean }) {
  // When workflow is done, show checkmarks instead of spinners
  let icon = MESSAGE_ICONS[message.type] || MESSAGE_ICONS[message.event || ''] || (
    <Loader2 size={12} className="text-gray-400" />
  )
  
  // Replace spinners with checkmarks when workflow is complete
  if (!isRunning) {
    const spinnerTypes = ['progress', 'stage_start', 'execution_start', 'rag_search_start']
    if (spinnerTypes.includes(message.type) || spinnerTypes.includes(message.event || '')) {
      icon = <CheckCircle2 size={12} className="text-green-500" />
    }
  }
  
  const text = getMessageText(message)
  const time = message.timestamp ? formatTimestamp(message.timestamp) : ''

  return (
    <div className="flex items-start gap-2 py-1 px-2 hover:bg-gray-50 rounded text-xs">
      <span className="mt-0.5 flex-shrink-0">{icon}</span>
      <span className="flex-1 text-gray-700 break-words">{text}</span>
      <span className="text-gray-400 flex-shrink-0">{time}</span>
    </div>
  )
}

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export function WorkflowProgressPanel({
  stage,
  progress,
  isRunning,
  wsMessages,
  startedAt,
  className,
}: WorkflowProgressPanelProps) {
  const [showActivity, setShowActivity] = useState(true)
  const [elapsed, setElapsed] = useState('0s')

  // Calculate elapsed time
  useEffect(() => {
    if (!startedAt) return

    // If not running, calculate final elapsed from last message or current time
    if (!isRunning) {
      // Use last message timestamp if available, otherwise current time
      const lastMsg = wsMessages[wsMessages.length - 1]
      const endTime = lastMsg?.timestamp ? new Date(lastMsg.timestamp).getTime() : Date.now()
      const duration = Math.floor((endTime - new Date(startedAt).getTime()) / 1000)
      if (duration < 60) {
        setElapsed(`${duration}s`)
      } else {
        const mins = Math.floor(duration / 60)
        const secs = duration % 60
        setElapsed(`${mins}m ${secs}s`)
      }
      return
    }

    // While running, update every second
    setElapsed(formatElapsed(startedAt))
    const interval = setInterval(() => {
      setElapsed(formatElapsed(startedAt))
    }, 1000)

    return () => clearInterval(interval)
  }, [isRunning, startedAt, wsMessages])

  // Filter relevant messages for activity log (skip pongs, etc.)
  const activityMessages = useMemo(() => {
    return wsMessages
      .filter((m) => m.type !== 'pong' && m.type !== 'connected' && m.type !== 'subscribed')
      .slice(-20) // Last 20 messages
  }, [wsMessages])

  // Current stage description
  const currentStageInfo = STAGES.find((s) => s.key === stage)

  if (!isRunning && wsMessages.length === 0) {
    return null
  }

  return (
    <div className={clsx('border-b border-gray-200 bg-white', className)}>
      {/* Header with progress bar */}
      <div className="px-4 py-3">
        {/* Stage pipeline */}
        <div className="flex items-center justify-center gap-0 mb-3 flex-wrap">
          {STAGES.filter((s) => s.key !== 'answer' || stage === 'answer').map((s, i, arr) => (
            <StageIndicator
              key={s.key}
              stage={s}
              status={getStageStatus(s.key, stage, wsMessages)}
              isLast={i === arr.length - 1}
            />
          ))}
        </div>

        {/* Progress bar */}
        <div className="relative h-2 bg-gray-100 rounded-full overflow-hidden mb-2">
          <div
            className="absolute inset-y-0 left-0 bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300"
            style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
          />
          {isRunning && (
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer" />
          )}
        </div>

        {/* Status line */}
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-2 text-gray-600">
            {isRunning ? (
              <>
                <Loader2 size={12} className="animate-spin text-blue-500" />
                <span className="font-medium">{currentStageInfo?.description || 'Processing...'}</span>
              </>
            ) : (
              <>
                <CheckCircle2 size={12} className="text-green-500" />
                <span className="font-medium text-green-600">Complete</span>
              </>
            )}
          </div>
          <div className="flex items-center gap-3 text-gray-400">
            <span>{Math.round(progress)}%</span>
            <div className="flex items-center gap-1">
              <Clock size={12} />
              <span>{elapsed}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Activity log toggle */}
      <button
        onClick={() => setShowActivity(!showActivity)}
        className="w-full flex items-center justify-between px-4 py-2 bg-gray-50 hover:bg-gray-100 text-xs text-gray-600 border-t border-gray-100"
      >
        <span className="flex items-center gap-1">
          {showActivity ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          Activity Log ({activityMessages.length} events)
        </span>
        {isRunning && activityMessages.length > 0 && (
          <span className="text-blue-500">Live</span>
        )}
      </button>

      {/* Activity log content */}
      {showActivity && activityMessages.length > 0 && (
        <div className="max-h-40 overflow-y-auto border-t border-gray-100">
          {activityMessages.map((msg, i) => (
            <ActivityItem key={`${msg.timestamp}-${i}`} message={msg} isRunning={isRunning} />
          ))}
        </div>
      )}

      {/* Shimmer animation style */}
      <style>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
        .animate-shimmer {
          animation: shimmer 1.5s infinite;
        }
      `}</style>
    </div>
  )
}

export default WorkflowProgressPanel