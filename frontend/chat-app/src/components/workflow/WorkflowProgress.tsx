import clsx from 'clsx'

interface WorkflowProgressProps {
  stage: string
  progress: number
  className?: string
}

const STAGE_LABELS: Record<string, string> = {
  starting: 'Starting...',
  gather_context: 'Gathering context',
  plan_and_decide: 'Planning analysis',
  plan: 'Planning',
  execute: 'Executing code',
  analyze_plots: 'Analyzing plots',
  answer: 'Generating answer',
  summarize: 'Summarizing',
  complete: 'Complete',
}

export function WorkflowProgress({ stage, progress, className }: WorkflowProgressProps) {
  const label = STAGE_LABELS[stage] || stage

  return (
    <div className={clsx('w-full', className)}>
      <div className="flex items-center justify-between text-sm mb-1">
        <span className="text-gray-600 font-medium">{label}</span>
        <span className="text-gray-400">{Math.round(progress)}%</span>
      </div>
      
      <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
        <div
          className="h-full gradient-bg transition-all duration-300 ease-out"
          style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
        />
      </div>
    </div>
  )
}

export default WorkflowProgress