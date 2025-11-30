# Complete Setup Guide: Running the Pipeline and Viewing Data in Grafana

This guide will walk you through **everything** from start to finish to get your monitoring system working with data in Grafana.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Start All Services](#step-1-start-all-services)
3. [Step 2: Verify Services Are Running](#step-2-verify-services-are-running)
4. [Step 3: Check API is Working](#step-3-check-api-is-working)
5. [Step 4: Generate Metrics by Making API Requests](#step-4-generate-metrics-by-making-api-requests)
6. [Step 5: Verify Prometheus is Scraping](#step-5-verify-prometheus-is-scraping)
7. [Step 6: Check Grafana Dashboard](#step-6-check-grafana-dashboard)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before starting, ensure you have:

- ✅ Docker Desktop installed and running
- ✅ Docker Compose installed
- ✅ At least 8GB RAM available
- ✅ Internet connection (for downloading images and API calls)

**Check Docker is running:**
```bash
docker --version
docker-compose --version
docker ps
```

---

## Step 1: Start All Services

### 1.1 Navigate to Project Directory

```bash
cd D:\Sem7\MLOPs\Project\CoinGecko0-Cryptocurrency-Prediction-MLOps
```

### 1.2 Start All Services

```bash
docker-compose up -d
```

This will:
- Build the API service (first time only)
- Pull Prometheus and Grafana images
- Start all services in the background

**Expected output:**
```
[+] Building X/X
[+] Running X/X
 ✔ Container crypto-prediction-api    Started
 ✔ Container prometheus               Started
 ✔ Container grafana                  Started
 ✔ Container airflow-postgres         Started
 ✔ Container minio                    Started
 ... (other services)
```

**⏱️ Wait Time:** First build takes 5-10 minutes. Subsequent starts take 30-60 seconds.

### 1.3 Check Build Status (if building)

If you see "Building..." wait for it to complete. You can watch the logs:

```bash
# Watch API build logs
docker-compose logs -f api

# Or watch all logs
docker-compose logs -f
```

Press `Ctrl+C` to stop watching logs.

---

## Step 2: Verify Services Are Running

### 2.1 Check All Containers Status

```bash
docker-compose ps
```

**Expected output:** All services should show "Up" status:
```
NAME                    STATUS
crypto-prediction-api   Up X minutes
prometheus              Up X minutes
grafana                 Up X minutes
airflow-postgres        Up X minutes
minio                   Up X minutes
...
```

### 2.2 Check Individual Service Health

```bash
# Check API service
docker-compose logs api | tail -20

# Check Prometheus
docker-compose logs prometheus | tail -20

# Check Grafana
docker-compose logs grafana | tail -20
```

**Look for:**
- API: "Application startup complete" or "Uvicorn running on"
- Prometheus: "Server is ready to receive web requests"
- Grafana: "HTTP Server Listen" or "Starting Grafana"

### 2.3 Verify Ports Are Accessible

Open these URLs in your browser:

| Service | URL | What You Should See |
|---------|-----|---------------------|
| **API** | http://localhost:8000 | JSON with API info |
| **API Docs** | http://localhost:8000/docs | Swagger UI |
| **Prometheus** | http://localhost:9090 | Prometheus UI |
| **Grafana** | http://localhost:3000 | Grafana login page |

**If any URL doesn't work:**
- Check the service is running: `docker-compose ps`
- Check logs for errors: `docker-compose logs <service-name>`
- Wait 30-60 seconds for services to fully start

---

## Step 3: Check API is Working

### 3.1 Test API Health Endpoint

```bash
# Using curl (PowerShell)
curl http://localhost:8000/health

# Or using browser
# Open: http://localhost:8000/health
```

**Expected response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-29T...",
  "loaded_models": []
}
```

### 3.2 Test Metrics Endpoint

```bash
curl http://localhost:8000/metrics
```

**Expected output:** You should see Prometheus metrics (even if empty):
```
# HELP api_requests_total Total number of API requests
# TYPE api_requests_total counter
api_requests_total{endpoint="/health",method="GET",status="success"} 1.0
...
```

**If you see metrics, the API is working! ✅**

---

## Step 4: Generate Metrics by Making API Requests

**This is the crucial step!** Grafana shows no data because no API requests have been made yet. We need to generate traffic.

### 4.1 Make Health Check Requests (Simple)

```bash
# Make 10 health check requests
for ($i=1; $i -le 10; $i++) {
    curl http://localhost:8000/health
    Start-Sleep -Seconds 1
}
```

### 4.2 Make Prediction Requests (Generates More Metrics)

**Note:** Prediction requests require a trained model. If you don't have models yet, skip to Step 4.3.

```bash
# Make a prediction request
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{\"coin_id\": \"bitcoin\", \"hours_ahead\": 1, \"model_type\": \"random_forest\"}'
```

**If you get an error about missing models**, that's okay! The request still generates metrics.

### 4.3 Generate Continuous Traffic (Recommended)

Create a script to continuously make requests:

**PowerShell Script (`generate-traffic.ps1`):**
```powershell
# Generate traffic for 2 minutes
$endTime = (Get-Date).AddMinutes(2)
$requestCount = 0

Write-Host "Generating API traffic..."
Write-Host "Press Ctrl+C to stop early"

while ((Get-Date) -lt $endTime) {
    try {
        # Health check
        Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get | Out-Null
        
        # Try prediction (may fail if no model, but still generates metrics)
        try {
            $body = @{
                coin_id = "bitcoin"
                hours_ahead = 1
                model_type = "random_forest"
            } | ConvertTo-Json
            
            Invoke-RestMethod -Uri "http://localhost:8000/predict" `
                -Method Post `
                -ContentType "application/json" `
                -Body $body | Out-Null
        } catch {
            # Prediction failed (no model), but that's okay
        }
        
        $requestCount++
        Write-Host "Requests made: $requestCount" -NoNewline
        Write-Host "`r" -NoNewline
        
        Start-Sleep -Seconds 2
    } catch {
        Write-Host "`nError: $_"
        Start-Sleep -Seconds 5
    }
}

Write-Host "`nCompleted! Made $requestCount requests"
```

**Run the script:**
```bash
powershell -ExecutionPolicy Bypass -File generate-traffic.ps1
```

**Or manually make requests every few seconds:**
```bash
# Run this multiple times
curl http://localhost:8000/health
curl http://localhost:8000/
curl http://localhost:8000/models/status
```

### 4.4 Verify Metrics Are Being Generated

```bash
# Check metrics endpoint again
curl http://localhost:8000/metrics | findstr "api_requests_total"
```

You should see increasing numbers:
```
api_requests_total{endpoint="/health",method="GET",status="success"} 5.0
api_requests_total{endpoint="/",method="GET",status="success"} 3.0
```

**✅ If you see metrics with values > 0, you're generating data!**

---

## Step 5: Verify Prometheus is Scraping

### 5.1 Check Prometheus Targets

1. Open http://localhost:9090
2. Go to **Status** → **Targets** (in the top menu)
3. Look for `crypto-prediction-api` target

**Expected status:** Should show **UP** (green)

**If it shows DOWN:**
- Check API is running: `docker-compose ps api`
- Check API metrics endpoint: `curl http://localhost:8000/metrics`
- Check Prometheus logs: `docker-compose logs prometheus`

### 5.2 Query Metrics in Prometheus

1. In Prometheus UI, go to **Graph** tab
2. Try these queries:

```promql
# Total API requests
api_requests_total

# Request rate
rate(api_requests_total[5m])

# API latency
histogram_quantile(0.95, rate(api_request_latency_seconds_bucket[5m]))
```

3. Click **Execute**
4. You should see data points in the graph

**✅ If you see data in Prometheus, it's working!**

### 5.3 Check Prometheus is Scraping API

In Prometheus, check:
- **Status** → **Targets** → `crypto-prediction-api` should be **UP**
- **Last Scrape** should be recent (within last 15 seconds)
- **Scrape Duration** should be < 1 second

---

## Step 6: Check Grafana Dashboard

### 6.1 Login to Grafana

1. Open http://localhost:3000
2. Login with:
   - **Username:** `admin`
   - **Password:** `admin`
3. You'll be prompted to change password (optional - click "Skip" for now)

### 6.2 Verify Prometheus Datasource

1. Go to **Configuration** (gear icon) → **Data sources**
2. Click on **Prometheus**
3. Check:
   - **URL:** Should be `http://prometheus:9090`
   - **Status:** Should show "Data source is working"
4. Click **Save & Test**

**If it shows an error:**
- Check Prometheus is running: `docker-compose ps prometheus`
- Check network connectivity: `docker-compose exec grafana ping prometheus`

### 6.3 Open the Dashboard

1. Go to **Dashboards** (grid icon) → **Browse**
2. Find **"Cryptocurrency Prediction API - Monitoring Dashboard"**
3. Click to open it

### 6.4 Check Dashboard Shows Data

**If you see "No data":**

1. **Check time range:**
   - Top right corner, set to **"Last 15 minutes"** or **"Last 1 hour"**
   - Click **Apply**

2. **Check queries are working:**
   - Click on any panel
   - Click **Edit**
   - Check the query (should start with `api_requests_total` or similar)
   - Click **Run query** (play button)
   - You should see data points

3. **Refresh dashboard:**
   - Click refresh icon (top right)
   - Or wait for auto-refresh (every 10 seconds)

### 6.5 What You Should See

After making API requests, you should see:

- **API Request Rate:** Line graph showing requests per second
- **API Request Latency:** Line graph with P50, P95, P99 latency
- **Total API Requests:** Increasing bar chart
- **Active API Requests:** Current number (usually 0-2)
- **Model Prediction Rate:** (if you made prediction requests)
- **Data Drift Ratio:** (if drift detection is initialized)

**✅ If panels show data, you're done!**

---

## Troubleshooting

### Problem: Grafana Shows "No Data"

**Solutions:**

1. **Check time range:**
   - Set to "Last 15 minutes" or "Last 1 hour"
   - Make sure you made requests within that time

2. **Verify Prometheus has data:**
   - Go to http://localhost:9090
   - Query: `api_requests_total`
   - If no data, go back to Step 4 and generate traffic

3. **Check Prometheus is scraping:**
   - Prometheus → Status → Targets
   - `crypto-prediction-api` should be UP
   - If DOWN, check API logs: `docker-compose logs api`

4. **Verify API is generating metrics:**
   ```bash
   curl http://localhost:8000/metrics | findstr "api_requests_total"
   ```
   Should show metrics with values > 0

5. **Check Grafana datasource:**
   - Configuration → Data sources → Prometheus
   - Should show "Data source is working"
   - URL should be `http://prometheus:9090`

### Problem: API Service Won't Start

**Check logs:**
```bash
docker-compose logs api
```

**Common issues:**
- **Port 8000 already in use:** Stop other services using port 8000
- **Build failed:** Check Dockerfile errors, rebuild: `docker-compose build api`
- **Dependency errors:** Check `requirements-api.txt` is correct

**Fix:**
```bash
# Rebuild API service
docker-compose build --no-cache api

# Restart
docker-compose up -d api
```

### Problem: Prometheus Can't Scrape API

**Check:**
1. API is running: `docker-compose ps api`
2. API metrics endpoint works: `curl http://localhost:8000/metrics`
3. Network connectivity: `docker-compose exec prometheus ping api`

**Fix:**
```bash
# Restart both services
docker-compose restart api prometheus

# Check Prometheus targets again
# Go to http://localhost:9090 → Status → Targets
```

### Problem: No Metrics Being Generated

**Make sure you're making API requests:**
```bash
# Simple test
curl http://localhost:8000/health
curl http://localhost:8000/health
curl http://localhost:8000/health

# Check metrics
curl http://localhost:8000/metrics | findstr "api_requests_total"
```

Should show increasing numbers.

### Problem: Dashboard Not Loading

**Check:**
1. Dashboard JSON is valid:
   ```bash
   # Check JSON syntax
   python -m json.tool monitoring/grafana/dashboards/crypto-prediction-monitoring.json
   ```

2. Grafana can read dashboard:
   - Check Grafana logs: `docker-compose logs grafana`
   - Look for dashboard loading errors

3. Restart Grafana:
   ```bash
   docker-compose restart grafana
   ```

---

## Quick Verification Checklist

Run through this checklist to verify everything is working:

- [ ] All services are running: `docker-compose ps` shows all "Up"
- [ ] API responds: `curl http://localhost:8000/health` returns JSON
- [ ] Metrics endpoint works: `curl http://localhost:8000/metrics` shows metrics
- [ ] Made API requests: Generated traffic (Step 4)
- [ ] Prometheus target is UP: http://localhost:9090 → Status → Targets
- [ ] Prometheus has data: Query `api_requests_total` shows results
- [ ] Grafana datasource works: Configuration → Data sources → Prometheus → "Data source is working"
- [ ] Dashboard loads: Can see dashboard in Grafana
- [ ] Dashboard shows data: Panels display graphs/charts

**If all checked ✅, you're done!**

---

## Next Steps

Once you have data in Grafana:

1. **Customize Dashboard:** Edit panels, add new metrics
2. **Set Up Alerts:** Configure notification channels (Slack, email)
3. **Initialize Drift Detection:** Run drift detector initialization script
4. **Monitor Continuously:** Keep making API requests to see real-time data

---

## Summary

The key to seeing data in Grafana is:

1. ✅ **Start all services** (docker-compose up -d)
2. ✅ **Make API requests** to generate metrics (Step 4 - THIS IS CRITICAL!)
3. ✅ **Wait for Prometheus to scrape** (15-30 seconds)
4. ✅ **Check Grafana dashboard** with correct time range

**Most common issue:** Not making API requests, so no metrics are generated!

---

**Need more help?** Check the logs:
```bash
docker-compose logs api
docker-compose logs prometheus
docker-compose logs grafana
```

