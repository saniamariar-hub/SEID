/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
      },
      colors: {
        cyber: {
          50:  '#e0f9ff',
          100: '#b3f0ff',
          200: '#80e5ff',
          300: '#4dd9ff',
          400: '#00cfff',
          500: '#00b8e6',
          600: '#008fb3',
          700: '#006680',
          800: '#003d4d',
          900: '#00141a',
        },
      },
      boxShadow: {
        'neon': '0 0 8px rgba(0,207,255,0.45), 0 0 20px rgba(0,207,255,0.20)',
        'neon-red': '0 0 8px rgba(239,68,68,0.5), 0 0 20px rgba(239,68,68,0.2)',
        'neon-green': '0 0 8px rgba(34,197,94,0.5), 0 0 20px rgba(34,197,94,0.2)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4,0,0.6,1) infinite',
      },
    },
  },
  plugins: [],
}

