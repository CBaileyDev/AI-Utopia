import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Plain Vite + React (React 18). The prototype was a single-file UMD app;
// this is the same visual output restructured into ES modules.
// No router/state libs: the app is simple useState tab-switching, matching the
// prototype. Keep it that way for the faithful port.
export default defineConfig({
  plugins: [react()],
  server: { port: 5173, open: false },
})
