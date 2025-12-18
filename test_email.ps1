# Test Email Configuration Script for Windows PowerShell

param(
    [Parameter(Mandatory=$true)]
    [string]$AppUrl,
    
    [Parameter(Mandatory=$true)]
    [string]$TestEmail
)

Write-Host "🧪 Testing email configuration..." -ForegroundColor Yellow
Write-Host "📍 App URL: $AppUrl" -ForegroundColor Cyan
Write-Host "📧 Test Email: $TestEmail" -ForegroundColor Cyan
Write-Host ""

try {
    # Test email endpoint
    $response = Invoke-RestMethod -Uri "$AppUrl/auth/test-email?email=$TestEmail" -Method POST -ErrorAction Stop
    
    Write-Host "✅ Response received:" -ForegroundColor Green
    Write-Host "Status: $($response.status)" -ForegroundColor White
    Write-Host "Message: $($response.message)" -ForegroundColor White
    Write-Host "SMTP Configured: $($response.smtp_configured)" -ForegroundColor White
    
    if ($response.status -eq "success") {
        Write-Host ""
        Write-Host "🎉 Email configuration is working!" -ForegroundColor Green
        Write-Host "📧 Check your email inbox for the test message." -ForegroundColor Yellow
    } elseif ($response.status -eq "failed") {
        Write-Host ""
        Write-Host "⚠️ SMTP not configured." -ForegroundColor Yellow
        Write-Host "📖 Follow GMAIL_SETUP.md to configure email verification." -ForegroundColor Cyan
    } else {
        Write-Host ""
        Write-Host "❌ Email test failed." -ForegroundColor Red
        if ($response.help) {
            Write-Host "💡 Help: $($response.help)" -ForegroundColor Cyan
        }
    }
    
} catch {
    Write-Host "❌ Failed to connect to the API:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host ""
    Write-Host "🔍 Check if your app URL is correct and the service is running." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📖 For email setup instructions, see GMAIL_SETUP.md" -ForegroundColor Cyan