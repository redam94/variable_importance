import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import tailwindcss from '@tailwindcss/vite'

const API_TARGET = process.env.VITE_API_URL || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  
  server: {
    port: 5173,
    host: true,
    
    proxy: {
      // REST API proxy
      '/api': {
        target: API_TARGET,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      
      // WebSocket proxy with proper error handling
      '/ws': {
        target: API_TARGET.replace('http', 'ws'),
        ws: true,
        changeOrigin: true,
        
        // Handle WebSocket upgrade
        configure: (proxy, _options) => {
          // Suppress EPIPE errors - these happen when client disconnects
          proxy.on('error', (err, _req, res) => {
            if ((err as NodeJS.ErrnoException).code === 'EPIPE') {
              // Client disconnected, ignore
              return
            }
            console.error('Proxy error:', err.message)
            
            // Only send response if it's writable
            if (res && 'writeHead' in res && !res.headersSent) {
              res.writeHead(502, { 'Content-Type': 'text/plain' })
              res.end('Proxy error')
            }
          })
          
          // Log WebSocket connections in dev
          proxy.on('proxyReqWs', (_proxyReq, req, _socket) => {
            console.log(`[WS] Proxying: ${req.url}`)
          })
          
          // Handle socket errors
          proxy.on('open', (proxySocket) => {
            proxySocket.on('error', (err) => {
              if ((err as NodeJS.ErrnoException).code !== 'EPIPE') {
                console.error('WebSocket error:', err.message)
              }
            })
          })
          
          // Handle close gracefully
          proxy.on('close', () => {
            // Connection closed, nothing to do
          })
        },
      },
    },
  },
  
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})