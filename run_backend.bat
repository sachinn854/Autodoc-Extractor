@echo off
echo 🚀 Starting Backend Server
echo.

cd backend

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    echo ✅ Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo ❌ Virtual environment not found!
    echo Please create one first: python -m venv venv
    pause
    exit /b 1
)

REM Install/update dependencies
echo 📥 Checking dependencies...
pip install -r requirements.txt --quiet

echo.
echo 🔄 Starting backend server...
echo 📍 Backend: http://localhost:8001
echo 📖 API Docs: http://localhost:8001/docs
echo 🛑 Press Ctrl+C to stop
echo.

uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload