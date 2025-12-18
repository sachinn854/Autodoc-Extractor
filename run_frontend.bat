@echo off
echo 🚀 Starting Frontend Server
echo.

cd frontend

REM Install/update dependencies
echo 📥 Checking dependencies...
call npm install --silent

echo.
echo 🔄 Starting frontend development server...
echo 📍 Frontend: http://localhost:3000
echo 🛑 Press Ctrl+C to stop
echo.

npm run dev