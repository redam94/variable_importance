import { useState, useCallback, useRef } from 'react'
import { useChatStore } from '../stores/chatStore.ts'
import { documentsApi } from '../lib/api.ts'
import type { Document as DocType } from '../types/api.ts'
import {
  Upload,
  Link,
  FileText,
  Trash2,
  Loader2,
  CheckCircle,
  AlertCircle,
  Globe,
  RefreshCw,
} from 'lucide-react'
import clsx from 'clsx'
import { useEffect } from 'react'

type TabType = 'upload' | 'url' | 'manage'

export function DocumentsPage() {
  const { workflowId } = useChatStore()
  const [activeTab, setActiveTab] = useState<TabType>('upload')
  const [documents, setDocuments] = useState<DocType[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  
  // Upload state
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  // URL scrape state
  const [url, setUrl] = useState('')
  const [urlTitle, setUrlTitle] = useState('')
  
  const fetchDocuments = useCallback(async () => {
    try {
      const docs = await documentsApi.list(workflowId)
      setDocuments(docs)
    } catch {
      // Documents may not exist yet
      setDocuments([])
    }
  }, [workflowId])
  
  useEffect(() => {
    fetchDocuments()
  }, [fetchDocuments])
  
  const handleFileUpload = async (files: FileList) => {
    setError(null)
    setSuccess(null)
    
    for (const file of Array.from(files)) {
      setLoading(true)
      setUploadProgress(0)
      
      try {
        const response = await documentsApi.upload(
          file,
          workflowId,
          undefined,
          (progress) => setUploadProgress(progress)
        )
        
        setSuccess(`Uploaded "${response.title}" (${response.chunk_count} chunks)`)
        await fetchDocuments()
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Upload failed')
      } finally {
        setLoading(false)
        setUploadProgress(0)
      }
    }
  }
  
  const handleUrlScrape = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!url.trim()) return
    
    setLoading(true)
    setError(null)
    setSuccess(null)
    
    try {
      const response = await documentsApi.scrapeURL({
        url: url.trim(),
        workflow_id: workflowId,
        title: urlTitle.trim() || undefined,
      })
      
      setSuccess(`Scraped "${response.title}" (${response.chunk_count} chunks)`)
      setUrl('')
      setUrlTitle('')
      await fetchDocuments()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Scraping failed')
    } finally {
      setLoading(false)
    }
  }
  
  const handleDelete = async (title: string) => {
    if (!confirm(`Delete "${title}"?`)) return
    
    try {
      await documentsApi.delete(workflowId, title)
      setSuccess(`Deleted "${title}"`)
      await fetchDocuments()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Delete failed')
    }
  }
  
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files)
    }
  }
  
  const getFileIcon = (sourceType: string) => {
    switch (sourceType) {
      case 'pdf':
        return '📕'
      case 'url':
        return '🌐'
      case 'md':
        return '📝'
      case 'csv':
        return '📊'
      case 'json':
        return '📋'
      default:
        return '📄'
    }
  }
  
  return (
    <div className="p-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">📚 Documents</h1>
        <p className="mt-1 text-gray-600">
          Upload files or scrape URLs to build your knowledge base
        </p>
      </div>
      
      {/* Tabs */}
      <div className="flex gap-2 mb-6">
        {[
          { id: 'upload', label: 'Upload Files', icon: Upload },
          { id: 'url', label: 'Scrape URL', icon: Globe },
          { id: 'manage', label: 'Manage', icon: FileText },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as TabType)}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors',
              activeTab === tab.id
                ? 'gradient-bg text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            )}
          >
            <tab.icon size={18} />
            {tab.label}
          </button>
        ))}
      </div>
      
      {/* Alerts */}
      {error && (
        <div className="mb-6 flex items-center gap-3 p-4 bg-red-50 text-red-700 rounded-lg">
          <AlertCircle size={20} />
          <p>{error}</p>
          <button
            onClick={() => setError(null)}
            className="ml-auto text-red-500 hover:text-red-700"
          >
            ×
          </button>
        </div>
      )}
      
      {success && (
        <div className="mb-6 flex items-center gap-3 p-4 bg-green-50 text-green-700 rounded-lg">
          <CheckCircle size={20} />
          <p>{success}</p>
          <button
            onClick={() => setSuccess(null)}
            className="ml-auto text-green-500 hover:text-green-700"
          >
            ×
          </button>
        </div>
      )}
      
      {/* Upload Tab */}
      {activeTab === 'upload' && (
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            📤 Upload Documents
          </h2>
          <p className="text-sm text-gray-600 mb-6">
            Supported formats: PDF, TXT, MD, CSV, JSON
          </p>
          
          {/* Drop zone */}
          <div
            onDragOver={(e) => {
              e.preventDefault()
              setIsDragging(true)
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={clsx(
              'border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-colors',
              isDragging
                ? 'border-primary-500 bg-primary-50'
                : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
            )}
          >
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept=".pdf,.txt,.md,.csv,.json"
              onChange={(e) => e.target.files && handleFileUpload(e.target.files)}
              className="hidden"
            />
            
            {loading ? (
              <div>
                <Loader2 className="w-12 h-12 mx-auto text-primary-500 animate-spin mb-4" />
                <p className="text-gray-600">Uploading... {Math.round(uploadProgress)}%</p>
                <div className="mt-4 w-48 mx-auto bg-gray-200 rounded-full h-2">
                  <div
                    className="gradient-bg h-2 rounded-full transition-all"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            ) : (
              <>
                <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
                <p className="text-gray-700 font-medium">
                  Drop files here or click to browse
                </p>
                <p className="text-sm text-gray-500 mt-2">
                  Maximum file size: 10MB
                </p>
              </>
            )}
          </div>
        </div>
      )}
      
      {/* URL Tab */}
      {activeTab === 'url' && (
        <div className="card p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            🌐 Scrape URL
          </h2>
          <p className="text-sm text-gray-600 mb-6">
            Extract content from web pages and add to your knowledge base
          </p>
          
          <form onSubmit={handleUrlScrape} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                URL
              </label>
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com/documentation"
                className="input-field"
                required
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Title (optional)
              </label>
              <input
                type="text"
                value={urlTitle}
                onChange={(e) => setUrlTitle(e.target.value)}
                placeholder="Auto-detected from page"
                className="input-field"
              />
            </div>
            
            <button
              type="submit"
              disabled={loading || !url.trim()}
              className="btn-primary flex items-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Scraping...
                </>
              ) : (
                <>
                  <Link size={18} />
                  Scrape URL
                </>
              )}
            </button>
          </form>
        </div>
      )}
      
      {/* Manage Tab */}
      {activeTab === 'manage' && (
        <div className="card p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold text-gray-900">
              📋 Documents ({documents.length})
            </h2>
            <button
              onClick={fetchDocuments}
              className="btn-ghost flex items-center gap-2"
            >
              <RefreshCw size={16} />
              Refresh
            </button>
          </div>
          
          {documents.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No documents yet. Upload files or scrape URLs to get started.</p>
            </div>
          ) : (
            <div className="space-y-3">
              {documents.map((doc) => (
                <div
                  key={doc.title}
                  className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                >
                  <span className="text-2xl">{getFileIcon(doc.source_type)}</span>
                  
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-gray-900 truncate">
                      {doc.title}
                    </p>
                    <p className="text-sm text-gray-500">
                      {doc.chunk_count} chunks • {doc.content_length.toLocaleString()} chars
                      {doc.url && (
                        <a
                          href={doc.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="ml-2 text-primary-600 hover:underline"
                        >
                          🔗 Source
                        </a>
                      )}
                    </p>
                  </div>
                  
                  <button
                    onClick={() => handleDelete(doc.title)}
                    className="p-2 text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                    title="Delete"
                  >
                    <Trash2 size={18} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      
      {/* Workflow context */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg text-sm text-gray-500">
        Documents are associated with workflow: <code className="px-1 bg-gray-200 rounded">{workflowId}</code>
      </div>
    </div>
  )
}