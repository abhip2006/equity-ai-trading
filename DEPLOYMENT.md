# Deploying Equity-AI-Trading to Railway

This guide walks you through deploying the Equity AI Trading application to Railway.app.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Architecture Overview](#architecture-overview)
- [Step-by-Step Deployment](#step-by-step-deployment)
- [Configuration Reference](#configuration-reference)
- [Post-Deployment](#post-deployment)
- [Troubleshooting](#troubleshooting)
- [Cost Estimates](#cost-estimates)

---

## Prerequisites

### Required Accounts

1. **Railway.app Account** - Sign up at [railway.app](https://railway.app)
2. **GitHub Account** - For connecting your repository
3. **Alpaca Trading Account(s)** - Get paper trading credentials from [alpaca.markets](https://alpaca.markets)
4. **LLM Provider API Keys** - At least one:
   - OpenAI API Key ([platform.openai.com](https://platform.openai.com))
   - Anthropic API Key ([console.anthropic.com](https://console.anthropic.com))
   - Google API Key (for Gemini)
   - Other optional providers (DeepSeek, Perplexity, etc.)

### Local Tools

- Git (for pushing code)
- Railway CLI (optional but recommended): `npm i -g @railway/cli`

---

## Architecture Overview

Your deployment will consist of **3 Railway services**:

```
┌─────────────────────────────────────────────────────────┐
│                    Railway Project                       │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  ┌──────────────────┐  ┌────────────────┐  ┌─────────┐ │
│  │  Backend Service │  │ Frontend Service│  │  Cron   │ │
│  │   (FastAPI)      │  │  (React/Vite)   │  │ Service │ │
│  │                  │  │                 │  │         │ │
│  │ • REST API       │  │ • Dashboard UI  │  │ • Equity│ │
│  │ • WebSocket      │  │ • Static Files  │  │  Logger │ │
│  │ • SQLite Volume  │  │                 │  │ (5 min) │ │
│  └──────────────────┘  └────────────────┘  └─────────┘ │
│          ↑                      ↓                ↑       │
│          │                      │                │       │
│          └──────────────────────┴────────────────┘       │
│                     Shared Volume: /data                 │
└─────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Deployment

### Phase 1: Prepare Your Repository

#### 1.1 Push Code to GitHub

```bash
# If not already a git repository
cd /path/to/equity-ai-trading
git init
git add .
git commit -m "feat: Prepare for Railway deployment"

# Create GitHub repository and push
gh repo create equity-ai-trading --private --source=. --push
# Or use the GitHub web interface
```

#### 1.2 Verify Configuration Files

Ensure these files exist in your repository:

- ✅ `backend/railway.json` - Backend deployment config
- ✅ `dashboard/railway.json` - Frontend deployment config
- ✅ `.env.example` - Environment variable template
- ✅ `backend/cron_equity_logger.py` - Standalone cron script
- ✅ `.gitignore` - Excludes secrets and databases

---

### Phase 2: Create Railway Project

#### 2.1 Create New Project

1. Go to [railway.app/new](https://railway.app/new)
2. Click **"Deploy from GitHub repo"**
3. Authorize Railway to access your GitHub account
4. Select the `equity-ai-trading` repository

#### 2.2 Upgrade to Pro Plan (Recommended)

For production use, upgrade to Railway Pro ($20/month):

- Click your profile → **Billing**
- Select **Pro Plan** ($20/month includes $20 usage credit)
- This enables:
  - 50GB persistent volume storage
  - Higher resource limits
  - Better uptime

---

### Phase 3: Deploy Backend Service

#### 3.1 Create Backend Service

1. In your Railway project, click **"+ New"** → **"GitHub Repo"**
2. Select your `equity-ai-trading` repository
3. Railway will detect it as a Python project

#### 3.2 Configure Backend Service

**Service Settings:**

- **Name**: `backend` or `equity-ai-trading-backend`
- **Root Directory**: `backend/`
- **Watch Paths**: `backend/**`
- **Start Command**: Auto-detected from `railway.json`

**Add Persistent Volume:**

1. In the service, go to **Settings** → **Volumes**
2. Click **"+ New Volume"**
3. Configure:
   - **Mount Path**: `/data`
   - **Size**: `10 GB` (adjust based on needs)
4. Click **Create Volume**

#### 3.3 Add Environment Variables

Go to **Variables** tab and add all required variables from `.env.example`:

**Trading Credentials (Required):**
```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

**LLM Provider Keys (At least one required):**
```
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

**Battle Royale Model Accounts (Optional but recommended):**
```
ALPACA_GPT4_API_KEY=your_gpt4_alpaca_key
ALPACA_GPT4_SECRET_KEY=your_gpt4_alpaca_secret

ALPACA_CLAUDE_API_KEY=your_claude_alpaca_key
ALPACA_CLAUDE_SECRET_KEY=your_claude_alpaca_secret

ALPACA_GEMINI_API_KEY=your_gemini_alpaca_key
ALPACA_GEMINI_SECRET_KEY=your_gemini_alpaca_secret

ALPACA_DEEPSEEK_API_KEY=your_deepseek_alpaca_key
ALPACA_DEEPSEEK_SECRET_KEY=your_deepseek_alpaca_secret

ALPACA_QWEN_API_KEY=your_qwen_alpaca_key
ALPACA_QWEN_SECRET_KEY=your_qwen_alpaca_secret

ALPACA_GROK_API_KEY=your_grok_alpaca_key
ALPACA_GROK_SECRET_KEY=your_grok_alpaca_secret
```

**Backend Configuration:**
```
DATABASE_PATH=/data
ENVIRONMENT=production
LOG_LEVEL=INFO
RATE_LIMIT_RPM=60
CORS_ORIGINS=http://localhost:3000
```

> **Note**: We'll update `CORS_ORIGINS` after deploying the frontend

#### 3.4 Deploy Backend

1. Click **Deploy** (or wait for auto-deploy)
2. Monitor deployment logs in the **Deployments** tab
3. Wait for "Deployment successful" message (2-5 minutes)

#### 3.5 Get Backend URL

1. Go to **Settings** → **Networking**
2. Click **"Generate Domain"**
3. Copy the generated URL (e.g., `https://equity-ai-trading-backend-production.up.railway.app`)
4. **Save this URL** - you'll need it for the frontend and CORS

#### 3.6 Update CORS Configuration

1. Go back to **Variables** tab
2. Update the `CORS_ORIGINS` variable:
   ```
   CORS_ORIGINS=http://localhost:3000,https://your-frontend-url.railway.app
   ```
   (You'll add the frontend URL after deploying it)

3. Service will auto-redeploy with new CORS settings

#### 3.7 Verify Backend Deployment

Test the health check endpoint:

```bash
curl https://your-backend-url.railway.app/
```

Expected response:
```json
{
  "status": "healthy",
  "service": "Trading Dashboard API",
  "timestamp": "2025-01-06T..."
}
```

---

### Phase 4: Deploy Frontend Service

#### 4.1 Create Frontend Service

1. Click **"+ New"** → **"GitHub Repo"**
2. Select `equity-ai-trading` repository again
3. Railway will detect Node.js/Vite project

#### 4.2 Configure Frontend Service

**Service Settings:**

- **Name**: `frontend` or `equity-ai-trading-dashboard`
- **Root Directory**: `dashboard/`
- **Watch Paths**: `dashboard/**`
- **Build Command**: Auto-detected from `railway.json` (`npm install && npm run build`)
- **Start Command**: Auto-detected from `railway.json` (`npx serve -s build -l $PORT`)

#### 4.3 Add Frontend Environment Variables

Go to **Variables** tab and add:

```
VITE_API_URL=https://your-backend-url.railway.app
VITE_WS_URL=wss://your-backend-url.railway.app
VITE_USE_MOCK_DATA=false
```

> Replace `your-backend-url.railway.app` with your actual backend URL from Phase 3.5

#### 4.4 Deploy Frontend

1. Click **Deploy**
2. Monitor build logs (this may take 3-5 minutes)
3. Wait for successful deployment

#### 4.5 Get Frontend URL

1. Go to **Settings** → **Networking**
2. Click **"Generate Domain"**
3. Copy the frontend URL (e.g., `https://equity-ai-trading-dashboard.up.railway.app`)

#### 4.6 Update Backend CORS

Go back to the **backend service**:

1. Navigate to **Variables**
2. Update `CORS_ORIGINS`:
   ```
   CORS_ORIGINS=http://localhost:3000,https://your-frontend-url.railway.app
   ```
3. Backend will auto-redeploy

#### 4.7 Verify Frontend Deployment

1. Open your frontend URL in a browser
2. You should see the Trading Dashboard
3. Check that it's connecting to the backend (Dashboard should load data)

---

### Phase 5: Deploy Cron Service (Equity Logger)

#### 5.1 Create Cron Service

1. Click **"+ New"** → **"Cron Job"**
2. Select `equity-ai-trading` repository

#### 5.2 Configure Cron Service

**Service Settings:**

- **Name**: `equity-logger-cron`
- **Root Directory**: `backend/`
- **Schedule**: `*/5 * * * *` (every 5 minutes)
- **Start Command**: `python cron_equity_logger.py`

#### 5.3 Add Environment Variables

The cron service needs the same environment variables as the backend:

**Option 1: Share Variables (Recommended)**
1. Go to backend service → **Variables**
2. Click each variable → **"Share Variable"**
3. Select the cron service

**Option 2: Copy Variables Manually**
Copy all Alpaca and LLM provider keys from backend to cron service.

**Additional Variables for Cron:**
```
DATABASE_PATH=/data
PYTHONPATH=/app
```

#### 5.4 Mount Volume (Critical!)

The cron service needs access to the same SQLite database:

1. Go to **Settings** → **Volumes**
2. **Attach the same volume** you created for the backend:
   - Select existing volume
   - Mount path: `/data`

> **Important**: Both backend and cron must use the same volume to share the database!

#### 5.5 Deploy Cron Service

1. Click **Deploy**
2. Monitor logs to verify it's working
3. Every 5 minutes, you should see equity snapshots being logged

#### 5.6 Verify Cron Execution

Check the logs after ~5 minutes:

```
Equity Logger Cron Job Starting
============================================================
✓ Logged gpt4: $100,234.56 (3 positions)
✓ Logged claude: $99,876.23 (2 positions)
Equity snapshot complete: 2 logged, 4 skipped, 0 errors
```

---

## Configuration Reference

### Environment Variables Summary

| Variable | Service | Required | Description |
|----------|---------|----------|-------------|
| `ALPACA_API_KEY` | Backend, Cron | ✅ | Primary Alpaca API key |
| `ALPACA_SECRET_KEY` | Backend, Cron | ✅ | Primary Alpaca secret |
| `ALPACA_BASE_URL` | Backend, Cron | ✅ | Alpaca API URL (paper/live) |
| `OPENAI_API_KEY` | Backend, Cron | ⚠️* | OpenAI API key |
| `ANTHROPIC_API_KEY` | Backend, Cron | ⚠️* | Anthropic API key |
| `DATABASE_PATH` | Backend, Cron | ✅ | Volume mount path (`/data`) |
| `CORS_ORIGINS` | Backend | ✅ | Allowed frontend origins |
| `VITE_API_URL` | Frontend | ✅ | Backend API endpoint |
| `VITE_WS_URL` | Frontend | ✅ | Backend WebSocket endpoint |

*At least one LLM provider key is required

### Volume Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| Mount Path | `/data` | Both backend and cron |
| Recommended Size | 10-20 GB | Grows with trading history |
| Backup Frequency | Automatic | Railway handles incremental backups |

---

## Post-Deployment

### 1. Test Complete Workflow

#### Check Backend API
```bash
# Health check
curl https://your-backend.railway.app/

# Get LLM models
curl https://your-backend.railway.app/api/llm-models

# Get leaderboard
curl https://your-backend.railway.app/api/battle/alpaca-leaderboard
```

#### Check Frontend

1. Open dashboard in browser
2. Verify dashboard loads without errors
3. Check that LLM models are displayed
4. Verify equity curves are rendering
5. Test WebSocket connection (real-time updates)

#### Check Cron Logs

1. Go to cron service → **Deployments** → **View Logs**
2. Wait 5 minutes for next execution
3. Verify equity snapshots are being logged

### 2. Monitor Resource Usage

**Backend Service:**
- Check **Metrics** tab for CPU/RAM usage
- Typical usage: 0.5-1 vCPU, 1-2 GB RAM

**Frontend Service:**
- Minimal resources needed (static files)
- Typical usage: 0.1 vCPU, 256 MB RAM

**Storage:**
- Monitor volume usage in **Volumes** tab
- SQLite databases grow ~10-50 MB per day depending on activity

### 3. Set Up Monitoring (Optional)

**Railway Native:**
- Enable deployment notifications (Settings → Notifications)
- Set up healthcheck alerts

**External Monitoring:**

**Sentry (Error Tracking):**
```bash
# Add to backend environment variables
SENTRY_DSN=your_sentry_dsn
```

**Better Stack (Logging):**
```bash
# Add to backend environment variables
LOGTAIL_TOKEN=your_logtail_token
```

### 4. Custom Domain (Optional)

1. Go to frontend service → **Settings** → **Networking**
2. Click **"Custom Domain"**
3. Add your domain (e.g., `trading.yourdomain.com`)
4. Update DNS records as instructed
5. SSL certificate will be auto-provisioned

Update CORS in backend after adding custom domain:
```
CORS_ORIGINS=https://trading.yourdomain.com
```

---

## Troubleshooting

### Backend Issues

**❌ "500 Internal Server Error"**
- Check logs for Python errors
- Verify all environment variables are set correctly
- Ensure volume is mounted at `/data`

**❌ "Database file not found"**
```bash
# Check DATABASE_PATH is set to /data
# Verify volume is attached and mounted
```

**❌ "Alpaca authentication failed"**
```bash
# Verify API keys in Railway dashboard
# Check keys are for paper trading mode
# Ensure no extra spaces in keys
```

### Frontend Issues

**❌ "Failed to fetch"**
- Check `VITE_API_URL` is set correctly
- Verify backend is deployed and healthy
- Check CORS is configured with frontend URL

**❌ "WebSocket connection failed"**
- Verify `VITE_WS_URL` uses `wss://` protocol
- Check backend WebSocket endpoint is working:
  ```bash
  curl -i -N \
    -H "Connection: Upgrade" \
    -H "Upgrade: websocket" \
    -H "Host: your-backend.railway.app" \
    -H "Origin: https://your-frontend.railway.app" \
    https://your-backend.railway.app/ws/llm-models
  ```

### Cron Issues

**❌ "Cron not executing"**
- Check cron schedule is valid: `*/5 * * * *`
- Verify logs show cron is running
- Check environment variables are set

**❌ "No equity data logged"**
- Verify Alpaca model API keys are configured
- Check volume is mounted correctly
- Ensure `DATABASE_PATH=/data`

### General Issues

**❌ "Out of memory"**
- Upgrade to larger Railway plan
- Optimize code to reduce memory usage

**❌ "Volume full"**
- Increase volume size in Railway dashboard
- Archive old database snapshots

**❌ "Rate limited"**
- Reduce API call frequency
- Implement additional caching
- Upgrade Alpaca account plan

---

## Cost Estimates

### Railway Pro Plan

**Monthly Costs:**

```
Base Subscription:           $20.00 (includes $20 usage credit)

Backend Service:
  - 1 vCPU average:          ~$20.00/month
  - 2 GB RAM:                ~$20.00/month

Frontend Service:
  - 0.1 vCPU average:        ~$ 2.00/month

Cron Service:
  - Minimal (runs 5 min):    ~$ 1.00/month

Volume (10 GB):              ~$ 1.50/month
Network Egress (100 GB):     ~$ 5.00/month

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Monthly Cost:          ~$69.50/month
Less included credit:        -$20.00
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Estimated Bill:              ~$49.50/month
```

**Cost Optimization Tips:**

1. **Use Trial Credit First**: Get $5 free for testing (30 days)
2. **Optimize Background Tasks**: Reduce cron frequency if not needed
3. **Monitor Usage**: Check Railway dashboard weekly
4. **Scale Down Dev Environments**: Use hobby plan for staging

### External Services (Optional)

- **Alpaca Paper Trading**: Free
- **LLM API Costs**: Variable (pay-per-use)
  - OpenAI GPT-4: ~$0.03/1K tokens (output)
  - Anthropic Claude: ~$0.015/1K tokens (output)
  - Google Gemini: Free tier available
- **Sentry (Error Tracking)**: Free tier (5K errors/month)
- **Better Stack (Logging)**: Free tier (10M logs/month)

---

## Next Steps

- ✅ Application deployed to Railway
- ✅ Backend API running with SQLite volume
- ✅ Frontend dashboard accessible
- ✅ Equity logging running every 5 minutes

**Recommended Next Steps:**

1. **Set up monitoring alerts** (Sentry, Better Stack)
2. **Configure custom domain** for frontend
3. **Enable automatic backups** of SQLite databases
4. **Test Battle Royale** with all 6 LLM models
5. **Monitor trading performance** via dashboard
6. **Review Railway metrics** weekly for optimization

---

## Support & Resources

- **Railway Documentation**: [docs.railway.app](https://docs.railway.app)
- **Railway Community**: [Discord](https://discord.gg/railway)
- **Alpaca API Docs**: [alpaca.markets/docs](https://alpaca.markets/docs)
- **Project Issues**: [GitHub Issues](https://github.com/yourusername/equity-ai-trading/issues)

---

**Congratulations! Your Equity AI Trading application is now live on Railway!** 🚀

You can now access your trading dashboard from anywhere and let your LLM traders compete 24/7.
