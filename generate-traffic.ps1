# Generate API Traffic Script
# This script makes continuous API requests to generate metrics for Grafana

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "API Traffic Generator" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$durationMinutes = 2
$requestInterval = 2  # seconds between requests
$apiUrl = "http://localhost:8000"

# Check if API is running
Write-Host "Checking API availability..." -ForegroundColor Yellow
try {
    $healthCheck = Invoke-RestMethod -Uri "$apiUrl/health" -Method Get -TimeoutSec 5
    Write-Host "[OK] API is running!" -ForegroundColor Green
    Write-Host "  Status: $($healthCheck.status)" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] API is not responding!" -ForegroundColor Red
    Write-Host "  Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please start the API service first:" -ForegroundColor Yellow
    Write-Host "  docker-compose up -d api" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Starting traffic generation..." -ForegroundColor Cyan
Write-Host "  Duration: $durationMinutes minutes" -ForegroundColor Cyan
Write-Host "  Interval: $requestInterval seconds between requests" -ForegroundColor Cyan
Write-Host "  Press Ctrl+C to stop early" -ForegroundColor Yellow
Write-Host ""

$endTime = (Get-Date).AddMinutes($durationMinutes)
$requestCount = 0
$successCount = 0
$errorCount = 0

while ((Get-Date) -lt $endTime) {
    try {
        # Health check request
        $healthResponse = Invoke-RestMethod -Uri "$apiUrl/health" -Method Get -TimeoutSec 5
        $successCount++
        
        # Try prediction request (may fail if no model, but still generates metrics)
        try {
            $body = @{
                coin_id = "bitcoin"
                hours_ahead = 1
                model_type = "random_forest"
            } | ConvertTo-Json
            
            $predictionResponse = Invoke-RestMethod -Uri "$apiUrl/predict" `
                -Method Post `
                -ContentType "application/json" `
                -Body $body `
                -TimeoutSec 10
            $successCount++
        } catch {
            # Prediction failed (no model), but that's okay - still generates metrics
            $successCount++
        }
        
        # Root endpoint
        $rootResponse = Invoke-RestMethod -Uri "$apiUrl/" -Method Get -TimeoutSec 5
        $successCount++
        
        # Models status
        try {
            $modelsResponse = Invoke-RestMethod -Uri "$apiUrl/models/status" -Method Get -TimeoutSec 5
            $successCount++
        } catch {
            $successCount++
        }
        
        $requestCount += 4
        
        # Display progress
        $remaining = ($endTime - (Get-Date)).TotalSeconds
        $progress = [math]::Round((1 - ($remaining / ($durationMinutes * 60))) * 100, 1)
        Write-Host "`rRequests: $requestCount | Success: $successCount | Errors: $errorCount | Progress: $progress% | Time remaining: $([math]::Round($remaining))s" -NoNewline -ForegroundColor Green
        
        Start-Sleep -Seconds $requestInterval
    } catch {
        $errorCount++
        Write-Host "`nError: $_" -ForegroundColor Red
        Start-Sleep -Seconds 5
    }
}

Write-Host ""
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Traffic Generation Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Total Requests: $requestCount" -ForegroundColor Cyan
Write-Host "Successful: $successCount" -ForegroundColor Green
Write-Host "Errors: $errorCount" -ForegroundColor $(if ($errorCount -gt 0) { "Red" } else { "Green" })
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Check Prometheus: http://localhost:9090" -ForegroundColor Yellow
Write-Host "2. Check Grafana: http://localhost:3000" -ForegroundColor Yellow
Write-Host "3. View dashboard: Dashboards -> Cryptocurrency Prediction API" -ForegroundColor Yellow
Write-Host ""

