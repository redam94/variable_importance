import { RefreshCw, CheckCircle, XCircle, Circle } from 'lucide-react'
import clsx from 'clsx'

export interface WorkflowStage {
  name: string
  status: 'pending' | 'running' | 'completed' | 'error'
  message?: string
}

interface WorkflowProgressProps {
  stages: WorkflowStage[]
  currentStage: string | null
  progress: number
  isRunning: boolean
}

const STAGE_LABELS: Record<string, string> = {
  gather_context: 'Gathering Context',
  plan_and_decide: 'Planning',
  execute: 'Executing Code',
  answer_from_context: 'Answering',
  summarize: 'Summarizing',
}

interface WorkflowStageItemProps {
  stage: WorkflowStage
}

export function WorkflowStageItem({ stage }: WorkflowStageItemProps) {
  return (
    <div
      className={clsx(
        'flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium',
        stage.status === 'completed' && 'bg-green-100 text-green-700',
        stage.status === 'running' && 'bg-blue-100 text-blue-700',
        stage.status === 'error' && 'bg-red-100 text-red-700',
        stage.status === 'pending' && 'bg-gray-100 text-gray-500'
      )}
    >
      {stage.status === 'running' && <RefreshCw size={12} className="animate-spin" />}
      {stage.status === 'completed' && <CheckCircle size={12} />}
      {stage.status === 'error' && <XCircle size={12} />}
      {stage.status === 'pending' && <Circle size={12} />}
      {STAGE_LABELS[stage.name] || stage.name}
    </div>
  )
}

export function WorkflowProgress({ stages, currentStage, progress, isRunning }: WorkflowProgressProps) {
  if (!isRunning && stages.length === 0) return null

  return (
    <div className="bg-gray-50 rounded-lg p-4 mb-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-gray-700">Workflow Progress</span>
        <span className="text-sm text-gray-500">{Math.round(progress)}%</span>
      </div>

      {/* Progress bar */}
      <div className="h-2 bg-gray-200 rounded-full mb-4 overflow-hidden">
        <div
          className="h-full gradient-bg transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Stage indicators */}
      <div className="flex gap-2 flex-wrap">
        {stages.map((stage) => (
          <WorkflowStageItem key={stage.name} stage={stage} />
        ))}
      </div>

      {/* Current stage message */}
      {currentStage && (
        <p className="mt-3 text-xs text-gray-500 flex items-center gap-1.5">
          <RefreshCw size={12} className="animate-spin" />
          {STAGE_LABELS[currentStage] || currentStage}...
        </p>
      )}
    </div>
  )
}