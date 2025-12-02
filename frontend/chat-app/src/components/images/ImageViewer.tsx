import { useState, useEffect, useCallback } from 'react'
import { X, ChevronLeft, ChevronRight, Download, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react'
import clsx from 'clsx'
import type { ImageInfo } from '../../types/api'
import { getStoredToken } from '../../lib/api'

const API_BASE = '/api'

interface ImageViewerProps {
  images: ImageInfo[]
  initialIndex?: number
  onClose: () => void
}

/**
 * Full-screen image viewer with navigation and zoom
 */
export function ImageViewer({ images, initialIndex = 0, onClose }: ImageViewerProps) {
  const [currentIndex, setCurrentIndex] = useState(initialIndex)
  const [zoom, setZoom] = useState(1)
  const [imageSrc, setImageSrc] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const currentImage = images[currentIndex]

  // Fetch image with auth
  const loadImage = useCallback(async (image: ImageInfo) => {
    setLoading(true)
    setError(null)
    
    try {
      const token = getStoredToken()
      const response = await fetch(`${API_BASE}${image.url}`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      })
      
      if (!response.ok) {
        throw new Error(`Failed to load image: ${response.statusText}`)
      }
      
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)
      setImageSrc(url)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load image')
    } finally {
      setLoading(false)
    }
  }, [])

  // Load image when index changes
  useEffect(() => {
    if (currentImage) {
      loadImage(currentImage)
    }
    
    return () => {
      if (imageSrc) {
        URL.revokeObjectURL(imageSrc)
      }
    }
  }, [currentImage, loadImage])

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      switch (e.key) {
        case 'Escape':
          onClose()
          break
        case 'ArrowLeft':
          setCurrentIndex((i) => (i > 0 ? i - 1 : images.length - 1))
          break
        case 'ArrowRight':
          setCurrentIndex((i) => (i < images.length - 1 ? i + 1 : 0))
          break
        case '+':
        case '=':
          setZoom((z) => Math.min(z + 0.25, 4))
          break
        case '-':
          setZoom((z) => Math.max(z - 0.25, 0.25))
          break
        case '0':
          setZoom(1)
          break
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [images.length, onClose])

  const handleDownload = async () => {
    if (!imageSrc || !currentImage) return
    
    const a = document.createElement('a')
    a.href = imageSrc
    a.download = currentImage.filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  if (images.length === 0) return null

  return (
    <div className="fixed inset-0 z-50 bg-black/90 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 bg-black/50">
        <div className="text-white">
          <span className="font-medium">{currentImage?.filename}</span>
          <span className="text-gray-400 ml-3 text-sm">
            {currentIndex + 1} / {images.length}
          </span>
          <span className="text-gray-500 ml-3 text-xs">
            {currentImage?.stage}
          </span>
        </div>
        
        <div className="flex items-center gap-2">
          {/* Zoom controls */}
          <button
            onClick={() => setZoom((z) => Math.max(z - 0.25, 0.25))}
            className="p-2 text-gray-400 hover:text-white transition-colors"
            title="Zoom out (-)"
          >
            <ZoomOut size={20} />
          </button>
          <span className="text-gray-400 text-sm w-14 text-center">
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={() => setZoom((z) => Math.min(z + 0.25, 4))}
            className="p-2 text-gray-400 hover:text-white transition-colors"
            title="Zoom in (+)"
          >
            <ZoomIn size={20} />
          </button>
          <button
            onClick={() => setZoom(1)}
            className="p-2 text-gray-400 hover:text-white transition-colors"
            title="Reset zoom (0)"
          >
            <Maximize2 size={20} />
          </button>
          
          <div className="w-px h-6 bg-gray-700 mx-2" />
          
          {/* Download */}
          <button
            onClick={handleDownload}
            className="p-2 text-gray-400 hover:text-white transition-colors"
            title="Download"
          >
            <Download size={20} />
          </button>
          
          {/* Close */}
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-white transition-colors"
            title="Close (Esc)"
          >
            <X size={24} />
          </button>
        </div>
      </div>

      {/* Image container */}
      <div className="flex-1 relative overflow-auto flex items-center justify-center p-4">
        {loading && (
          <div className="text-white">Loading...</div>
        )}
        
        {error && (
          <div className="text-red-400">{error}</div>
        )}
        
        {!loading && !error && imageSrc && (
          <img
            src={imageSrc}
            alt={currentImage?.filename}
            className="max-w-none transition-transform duration-200"
            style={{ transform: `scale(${zoom})` }}
            draggable={false}
          />
        )}

        {/* Navigation arrows */}
        {images.length > 1 && (
          <>
            <button
              onClick={() => setCurrentIndex((i) => (i > 0 ? i - 1 : images.length - 1))}
              className="absolute left-4 top-1/2 -translate-y-1/2 p-3 bg-black/50 rounded-full text-white hover:bg-black/70 transition-colors"
              title="Previous (←)"
            >
              <ChevronLeft size={28} />
            </button>
            <button
              onClick={() => setCurrentIndex((i) => (i < images.length - 1 ? i + 1 : 0))}
              className="absolute right-4 top-1/2 -translate-y-1/2 p-3 bg-black/50 rounded-full text-white hover:bg-black/70 transition-colors"
              title="Next (→)"
            >
              <ChevronRight size={28} />
            </button>
          </>
        )}
      </div>

      {/* Thumbnail strip */}
      {images.length > 1 && (
        <div className="bg-black/50 px-4 py-3 overflow-x-auto">
          <div className="flex gap-2 justify-center">
            {images.map((img, idx) => (
              <ImageThumbnail
                key={img.path}
                image={img}
                isActive={idx === currentIndex}
                onClick={() => setCurrentIndex(idx)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// =============================================================================
// Thumbnail component
// =============================================================================

interface ImageThumbnailProps {
  image: ImageInfo
  isActive: boolean
  onClick: () => void
}

function ImageThumbnail({ image, isActive, onClick }: ImageThumbnailProps) {
  const [src, setSrc] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    
    const loadThumb = async () => {
      try {
        const token = getStoredToken()
        const response = await fetch(`${API_BASE}${image.url}`, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        })
        if (!response.ok || cancelled) return
        
        const blob = await response.blob()
        if (!cancelled) {
          setSrc(URL.createObjectURL(blob))
        }
      } catch {
        // Ignore thumbnail errors
      }
    }

    loadThumb()
    
    return () => {
      cancelled = true
      if (src) URL.revokeObjectURL(src)
    }
  }, [image.url])

  return (
    <button
      onClick={onClick}
      className={clsx(
        'w-16 h-16 rounded overflow-hidden border-2 transition-colors flex-shrink-0',
        isActive ? 'border-blue-500' : 'border-transparent hover:border-gray-500'
      )}
    >
      {src ? (
        <img
          src={src}
          alt={image.filename}
          className="w-full h-full object-cover"
        />
      ) : (
        <div className="w-full h-full bg-gray-700 animate-pulse" />
      )}
    </button>
  )
}

// =============================================================================
// Image grid for displaying multiple images
// =============================================================================

interface ImageGridProps {
  images: ImageInfo[]
  onImageClick?: (index: number) => void
  className?: string
}

export function ImageGrid({ images, onImageClick, className }: ImageGridProps) {
  return (
    <div className={clsx('grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4', className)}>
      {images.map((image, idx) => (
        <ImageCard
          key={image.path}
          image={image}
          onClick={() => onImageClick?.(idx)}
        />
      ))}
    </div>
  )
}

interface ImageCardProps {
  image: ImageInfo
  onClick?: () => void
}

function ImageCard({ image, onClick }: ImageCardProps) {
  const [src, setSrc] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    
    const load = async () => {
      setLoading(true)
      try {
        const token = getStoredToken()
        const response = await fetch(`${API_BASE}${image.url}`, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        })
        if (!response.ok || cancelled) return
        
        const blob = await response.blob()
        if (!cancelled) {
          setSrc(URL.createObjectURL(blob))
        }
      } catch {
        // Ignore errors
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()
    
    return () => {
      cancelled = true
      if (src) URL.revokeObjectURL(src)
    }
  }, [image.url])

  const sizeKB = (image.size_bytes / 1024).toFixed(1)

  return (
    <div
      onClick={onClick}
      className={clsx(
        'group relative bg-gray-100 rounded-lg overflow-hidden cursor-pointer',
        'hover:ring-2 hover:ring-blue-500 transition-all'
      )}
    >
      <div className="aspect-square">
        {loading ? (
          <div className="w-full h-full bg-gray-200 animate-pulse" />
        ) : src ? (
          <img
            src={src}
            alt={image.filename}
            className="w-full h-full object-contain bg-white"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-400">
            Failed to load
          </div>
        )}
      </div>
      
      {/* Overlay with info */}
      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent p-2 opacity-0 group-hover:opacity-100 transition-opacity">
        <p className="text-white text-sm truncate">{image.filename}</p>
        <p className="text-gray-300 text-xs">{sizeKB} KB • {image.stage}</p>
      </div>
    </div>
  )
}