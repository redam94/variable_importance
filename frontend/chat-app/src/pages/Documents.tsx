import { useState, useCallback, useRef, useEffect } from 'react'
import { useChatStore, selectCurrentWorkflowId } from '../stores/chatStore'
import { documentsApi } from '../lib/api'
import type { Document as DocType } from '../types/api'
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
  FolderOpen,
} from 'lucide-react'
import clsx from 'clsx'

type TabType = 'upload' | 'url' | 'manage'

export function DocumentsPage() {
  const workflowId = useChatStore(selectCurrentWorkflowId)
  const initSession = useChatStore((state) => state.initSession)
  
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
  
  // Init session on mount
  useEffect(() => {
    initSession('workflow-chat')
  }, [initSession])
  
  const fetchDocuments = useCallback(async () => {
    if (!workflowId) return
    try {
      const docs = await documentsApi.list(workflowId)
      setDocuments(docs)
    } catch {
      setDocuments([])
    }
  }, [workflowId])
  
  useEffect(() => {
    fetchDocuments()
  }, [fetchDocuments])
  
  const handleFileUpload = async (files: FileList) => {
    if (!workflowId) return
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
    if (!url.trim() || !workflowId) return
    
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
      setError(err instanceof Error ? err.message : 'Scrape failed')
    } finally {
      setLoading(false)
    }
  }
  
  const handleDelete = async (docId: string) => {
    if (!workflowId || !confirm('Delete this document?')) return
    
    try {
      await documentsApi.delete(docId, workflowId)
      setSuccess('Document deleted')
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
  
  return (
    <div className="p-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">📄 Documents</h1>
        <div className="flex items-center gap-2 mt-1 text-gray-600">
          <FolderOpen size={16} />
          <span>{workflowId || 'No workflow selected'}</span>
        </div>
      </div>
      
      {/* Alerts */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center gap-3 text-red-700">
          <AlertCircle size={20} />
          {error}
        </div>
      )}
      
      {success && (
        <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg flex items-center gap-3 text-green-700">
          <CheckCircle size={20} />
          {success}
        </div>
      )}
      
      {/* Tabs */}
      <div className="flex gap-2 mb-6">
        {[
          { id: 'upload' as TabType, label: 'Upload Files', icon: Upload },
          { id: 'url' as TabType, label: 'Add URL', icon: Globe },
          { id: 'manage' as TabType, label: 'Manage', icon: FileText },
        ].map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors',
              activeTab === id
                ? 'bg-primary-100 text-primary-700'
                : 'text-gray-600 hover:bg-gray-100'
            )}
          >
            <Icon size={18} />
            {label}
          </button>
        ))}
      </div>
      
      {/* Upload Tab */}
      {activeTab === 'upload' && (
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true) }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={clsx(
            'card p-12 border-2 border-dashed text-center transition-all',
            isDragging ? 'border-primary-400 bg-primary-50' : 'border-gray-300',
            loading && 'opacity-50 pointer-events-none'
          )}
        >
          <Upload className="w-12 h-12 mx-auto text-gray-400 mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            Drop files here or click to upload
          </h3>
          <p className="text-gray-500 mb-4">
            Supports PDF, TXT, MD, CSV, JSON files
          </p>
          
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.txt,.md,.csv,.json"
            onChange={(e) => e.target.files && handleFileUpload(e.target.files)}
            className="hidden"
          />
          
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={loading}
            className="btn-primary"
          >
            {loading ? (
              <>
                <Loader2 size={18} className="animate-spin mr-2" />
                Uploading {uploadProgress}%
              </>
            ) : (
              'Select Files'
            )}
          </button>
        </div>
      )}
      
      {/* URL Tab */}
      {activeTab === 'url' && (
        <form onSubmit={handleUrlScrape} className="card p-6">
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                URL to scrape
              </label>
              <div className="relative">
                <Link className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={18} />
                <input
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://example.com/article"
                  className="input-field pl-10"
                  required
                />
              </div>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Title (optional)
              </label>
              <input
                type="text"
                value={urlTitle}
                onChange={(e) => setUrlTitle(e.target.value)}
                placeholder="Custom title for this document"
                className="input-field"
              />
            </div>
            
            <button
              type="submit"
              disabled={loading || !url.trim()}
              className="btn-primary w-full"
            >
              {loading ? (
                <>
                  <Loader2 size={18} className="animate-spin mr-2" />
                  Scraping...
                </>
              ) : (
                <>
                  <Globe size={18} className="mr-2" />
                  Scrape URL
                </>
              )}
            </button>
          </div>
        </form>
      )}
      
      {/* Manage Tab */}
      {activeTab === 'manage' && (
        <div>
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-medium text-gray-900">
              {documents.length} Documents
            </h3>
            <button
              onClick={fetchDocuments}
              className="btn-ghost flex items-center gap-2"
            >
              <RefreshCw size={16} />
              Refresh
            </button>
          </div>
          
          {documents.length === 0 ? (
            <div className="card p-12 text-center">
              <FileText className="w-12 h-12 mx-auto text-gray-300 mb-4" />
              <p className="text-gray-600">No documents yet</p>
              <p className="text-sm text-gray-500 mt-2">
                Upload files or scrape URLs to get started
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {documents.map((doc) => (
                <div
                  key={doc.title}
                  className="card p-4 flex items-center justify-between"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-gray-100 flex items-center justify-center">
                      <FileText size={20} className="text-gray-500" />
                    </div>
                    <div>
                      <p className="font-medium text-gray-900">{doc.title}</p>
                      <p className="text-sm text-gray-500">
                        {doc.chunk_count} chunks • {doc.source_type}
                      </p>
                    </div>
                  </div>
                  
                  <button
                    onClick={() => handleDelete(doc.title)}
                    className="btn-ghost text-red-500 hover:bg-red-50"
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
        Adding documents to: <code className="px-1 bg-gray-200 rounded">{workflowId || 'None'}</code>
      </div>
    </div>
  )
}