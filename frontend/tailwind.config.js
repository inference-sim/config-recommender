/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: '#2563EB',
          50: '#EFF6FF',
          100: '#DBEAFE',
          600: '#2563EB',
          700: '#1D4ED8',
          800: '#1E40AF',
        },
        success: {
          DEFAULT: '#10B981',
          100: '#D1FAE5',
          600: '#10B981',
        },
        warning: {
          DEFAULT: '#F59E0B',
          100: '#FEF3C7',
          600: '#F59E0B',
        },
        error: {
          DEFAULT: '#EF4444',
          100: '#FEE2E2',
          600: '#EF4444',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
