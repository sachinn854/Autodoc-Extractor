@echo off
echo 🚀 Starting Autodoc Extractor - Local Development
echo.

REM Check if we're in the right directory
if not exist "backend" (
    echo ❌ Error: backend directory not found
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

if not exist "frontend" (
    echo ❌ Error: frontend directory not found
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

echo 📦 Setting up Backend...
cd backend

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    echo ✅ Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ⚠️ Virtual environment not found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo ✅ Virtual environment created and activated
)

REM Install backend dependencies
echo 📥 Installing backend dependencies...
pip install -r requirements.txt

REM Check if installation was successful
if %errorlevel% neq 0 (
    echo ❌ Failed to install backend dependencies
    pause
    exit /b 1
)

echo ✅ Backend dependencies installed successfully
echo.

REM Start backend server in background
echo 🔄 Starting backend server on http://localhost:8001
echo 📖 API Documentation: http://localhost:8001/docs
echo.
start "Backend Server" cmd /k "venv\Scripts\activate.bat && uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload"

REM Wait a moment for backend to start
timeout /t 3 /nobreak >nul

REM Go back to root and setup frontend
cd ..
echo 📦 Setting up Frontend...
cd frontend

REM Install frontend dependencies
echo 📥 Installing frontend dependencies...
call npm install

REM Check if installation was successful
if %errorlevel% neq 0 (
    echo ❌ Failed to install frontend dependencies
    pause
    exit /b 1
)

echo ✅ Frontend dependencies installed successfully
echo.

REM Start frontend server
echo 🔄 Starting frontend server on http://localhost:3000
echo.
start "Frontend Server" cmd /k "npm run dev"

echo.
echo 🎉 Both servers are starting up!
echo.
echo 📍 URLs:
echo   Frontend: http://localhost:3000
echo   Backend:  http://localhost:8001
echo   API Docs: http://localhost:8001/docs
echo.
echo 💡 Tip: Wait 10-15 seconds for servers to fully start
echo 🛑 To stop servers: Close the terminal windows or press Ctrl+C
echo.
pause