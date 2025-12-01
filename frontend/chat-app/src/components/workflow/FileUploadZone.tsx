import { useState, useRef, useCallback } from 'react'
import { Upload, FileSpreadsheet, X, CheckCircle, Loader2 } from 'lucide-react'
import clsx from 'clsx'

interface FileUploadZoneProps {
  onFileSelect: (file: File) => void
  accept?: string
  disabled?: boolean
  currentFile?: File | null
  uploadProgress?: number
  isUploading?: boolean
  uploadSuccess?: boolean
}

export function FileUploadZone({
  onFileSelect,
  accept = '.csv,.xlsx,.xls',
  disabled,
  currentFile,
  uploadProgress = 0,
  isUploading,
  uploadSuccess,
}: FileUploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    if (!disabled) setIsDragging(true)
  }, [disabled])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (disabled) return

    const file = e.dataTransfer.files[0]
    if (file) onFileSelect(file)
  }, [disabled, onFileSelect])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) onFileSelect(file)
  }

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={() => !disabled && fileInputRef.current?.click()}
      className={clsx(
        'relative border-2 border-dashed rounded-lg p-4 transition-colors cursor-pointer',
        isDragging && 'border-primary-500 bg-primary-50',
        disabled && 'opacity-50 cursor-not-allowed',
        currentFile ? 'border-green-300 bg-green-50' : 'border-gray-300 hover:border-gray-400',
      )}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept={accept}
        onChange={handleFileChange}
        disabled={disabled}
        className="hidden"
      />

      {currentFile ? (
        <div className="flex items-center gap-3">
          <div className={clsx(
            'p-2 rounded-lg',
            uploadSuccess ? 'bg-green-100 text-green-600' : 'bg-primary-100 text-primary-600'
          )}>
            {isUploading ? (
              <Loader2 size={20} className="animate-spin" />
            ) : uploadSuccess ? (
              <CheckCircle size={20} />
            ) : (
              <FileSpreadsheet size={20} />
            )}
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-gray-900 truncate">{currentFile.name}</p>
            <p className="text-xs text-gray-500">{formatFileSize(currentFile.size)}</p>
          </div>
          {!isUploading && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                onFileSelect(null as unknown as File) // Clear file
              }}
              className="p-1 hover:bg-gray-200 rounded text-gray-500"
            >
              <X size={16} />
            </button>
          )}
        </div>
      ) : (
        <div className="flex flex-col items-center text-center py-2">
          <Upload size={24} className="text-gray-400 mb-2" />
          <p className="text-sm text-gray-600">
            Drop a data file or <span className="text-primary-600 font-medium">browse</span>
          </p>
          <p className="text-xs text-gray-400 mt-1">CSV, Excel files supported</p>
        </div>
      )}

      {/* Upload progress */}
      {isUploading && uploadProgress > 0 && (
        <div className="absolute bottom-0 left-0 right-0 h-1 bg-gray-200 rounded-b-lg overflow-hidden">
          <div
            className="h-full bg-primary-500 transition-all"
            style={{ width: `${uploadProgress}%` }}
          />
        </div>
      )}
    </div>
  )
}