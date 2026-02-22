/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        slate: {
          800: '#1e293b',
          900: '#0f172a',
        },
        btc: '#f97316',
        eth: '#8b5cf6',
        ratio: '#3b82f6',
      },
    },
  },
  plugins: [],
}
