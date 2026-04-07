import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// `base: './'` makes all asset paths relative, so the site works at any
// subpath (e.g. https://<user>.github.io/llm_leaderboard/).
export default defineConfig({
  base: './',
  plugins: [react()],
})
