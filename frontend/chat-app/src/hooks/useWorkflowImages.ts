import { useState, useEffect, useCallback } from 'react'
import type { ImageInfo, WorkflowImages, StageFiles } from '../types/api'
import { filesApi } from '../lib/api'

interface UseWorkflowImagesOptions {
  workflowId: string
  stage?: string
  autoFetch?: boolean
}

interface UseWorkflowImagesReturn {
  images: ImageInfo[]
  totalImages: number
  loading: boolean
  error: string | null
  refetch: () => Promise<void>
}

/**
 * Hook to fetch and manage workflow images
 */
export function useWorkflowImages({
  workflowId,
  stage,
  autoFetch = true,
}: UseWorkflowImagesOptions): UseWorkflowImagesReturn {
  const [images, setImages] = useState<ImageInfo[]>([])
  const [totalImages, setTotalImages] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchImages = useCallback(async () => {
    if (!workflowId) return
    
    setLoading(true)
    setError(null)
    
    try {
      const result = await filesApi.listImages(workflowId, stage)
      setImages(result.images)
      setTotalImages(result.total_images)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch images')
      setImages([])
      setTotalImages(0)
    } finally {
      setLoading(false)
    }
  }, [workflowId, stage])

  useEffect(() => {
    if (autoFetch) {
      fetchImages()
    }
  }, [autoFetch, fetchImages])

  return {
    images,
    totalImages,
    loading,
    error,
    refetch: fetchImages,
  }
}

interface UseStageFilesReturn {
  stages: StageFiles[]
  loading: boolean
  error: string | null
  refetch: () => Promise<void>
}

/**
 * Hook to fetch all stages with their files
 */
export function useStageFiles(workflowId: string): UseStageFilesReturn {
  const [stages, setStages] = useState<StageFiles[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchStages = useCallback(async () => {
    if (!workflowId) return
    
    setLoading(true)
    setError(null)
    
    try {
      const result = await filesApi.listStagesWithFiles(workflowId)
      setStages(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch stages')
      setStages([])
    } finally {
      setLoading(false)
    }
  }, [workflowId])

  useEffect(() => {
    fetchStages()
  }, [fetchStages])

  return {
    stages,
    loading,
    error,
    refetch: fetchStages,
  }
}