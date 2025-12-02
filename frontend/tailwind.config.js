/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        credimed: {
          primary: '#55a4b1',
          secondary: '#f05454',
          dark: '#333333',
        }
      }
    },
  },
  plugins: [],
}