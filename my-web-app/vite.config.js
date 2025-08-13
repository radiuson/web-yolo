import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'



// https://vite.dev/config/
export default defineConfig({
  base: '/tokyo_expo_web/my-web-app/dist/',
  plugins: [react()],
})
