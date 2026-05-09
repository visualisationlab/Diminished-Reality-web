import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'

export default defineConfig({
  plugins: [react(), basicSsl()],
  server: {
    proxy: {
      '/ws': {
        target: 'ws://localhost:5174',
        ws: true,
      },
    },
  },
})
