# Railway Deployment Verification Checklist

Use this checklist to verify your deployment is working correctly.

## Pre-Deployment Checklist

- [ ] GitHub repository created and code pushed
- [ ] Railway account created (Pro plan recommended)
- [ ] Alpaca paper trading account(s) created
- [ ] LLM provider API keys obtained (at least one)
- [ ] All API keys saved securely (password manager recommended)

---

## Backend Service Checklist

- [ ] **Service Created**
  - [ ] Root directory set to `backend/`
  - [ ] Watch paths set to `backend/**`

- [ ] **Volume Configured**
  - [ ] Volume created (10+ GB recommended)
  - [ ] Mount path set to `/data`

- [ ] **Environment Variables Set** (copy from `.env.example`)
  - [ ] `ALPACA_API_KEY`
  - [ ] `ALPACA_SECRET_KEY`
  - [ ] `ALPACA_BASE_URL`
  - [ ] `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` (at least one)
  - [ ] `DATABASE_PATH=/data`
  - [ ] `ENVIRONMENT=production`
  - [ ] `LOG_LEVEL=INFO`
  - [ ] Battle model keys (optional): `ALPACA_GPT4_API_KEY`, etc.

- [ ] **Deployment Successful**
  - [ ] Build completed without errors
  - [ ] Service is running (green status)
  - [ ] Domain generated

- [ ] **Health Check Passes**
  ```bash
  curl https://your-backend-url.railway.app/
  ```
  Expected: `{"status": "healthy", ...}`

- [ ] **API Endpoints Working**
  - [ ] `/api/llm-models` returns model list
  - [ ] `/api/battle/alpaca-leaderboard` returns leaderboard
  - [ ] No CORS errors in browser console

---

## Frontend Service Checklist

- [ ] **Service Created**
  - [ ] Root directory set to `dashboard/`
  - [ ] Watch paths set to `dashboard/**`

- [ ] **Environment Variables Set**
  - [ ] `VITE_API_URL=https://your-backend-url.railway.app`
  - [ ] `VITE_WS_URL=wss://your-backend-url.railway.app`
  - [ ] `VITE_USE_MOCK_DATA=false`

- [ ] **Deployment Successful**
  - [ ] Vite build completed without errors
  - [ ] Service is running
  - [ ] Domain generated

- [ ] **Frontend Loads**
  - [ ] Dashboard opens in browser
  - [ ] No console errors
  - [ ] UI renders correctly
  - [ ] Data loads from backend

- [ ] **Features Working**
  - [ ] LLM models displayed
  - [ ] Leaderboard showing model rankings
  - [ ] Equity curves rendering
  - [ ] WebSocket connection active (real-time updates)

---

## Cron Service Checklist

- [ ] **Service Created**
  - [ ] Root directory set to `backend/`
  - [ ] Schedule set to `*/5 * * * *`
  - [ ] Start command: `python cron_equity_logger.py`

- [ ] **Volume Attached**
  - [ ] Same volume as backend attached
  - [ ] Mount path set to `/data`

- [ ] **Environment Variables Set**
  - [ ] All Alpaca model API keys configured
  - [ ] `DATABASE_PATH=/data`
  - [ ] `PYTHONPATH=/app`

- [ ] **Cron Executing**
  - [ ] First execution completed successfully
  - [ ] Logs show "Equity snapshot complete"
  - [ ] No errors in logs

- [ ] **Equity Data Logged**
  - [ ] Check backend endpoint: `/api/alpaca-equity-curves`
  - [ ] Data points appearing every 5 minutes
  - [ ] Equity curves showing in dashboard

---

## CORS Configuration Checklist

- [ ] **Backend CORS Updated**
  - [ ] `CORS_ORIGINS` includes frontend Railway URL
  - [ ] Backend redeployed after CORS update
  - [ ] No CORS errors in browser console

Example:
```
CORS_ORIGINS=https://your-frontend.railway.app,http://localhost:3000
```

---

## Security Checklist

- [ ] **Secrets Protected**
  - [ ] `.env` file NOT committed to git
  - [ ] `.env.production` NOT committed to git
  - [ ] All API keys set in Railway dashboard only

- [ ] **Optional: API Key Auth**
  - [ ] `API_KEY` environment variable set (if desired)
  - [ ] Frontend configured to send API key header

- [ ] **Rate Limiting**
  - [ ] `RATE_LIMIT_RPM` set (default: 60)
  - [ ] Test: Verify rate limiting blocks excessive requests

---

## Monitoring Checklist

- [ ] **Railway Metrics**
  - [ ] Backend CPU/RAM usage normal (< 80%)
  - [ ] Frontend CPU/RAM usage normal
  - [ ] Volume usage < 80%

- [ ] **Logs Review**
  - [ ] Backend logs show no critical errors
  - [ ] Cron logs show successful snapshots
  - [ ] No Alpaca rate limit errors

- [ ] **Optional: External Monitoring**
  - [ ] Sentry configured for error tracking
  - [ ] Better Stack configured for log aggregation

---

## Performance Testing Checklist

- [ ] **Backend Performance**
  ```bash
  # Test API response time
  time curl https://your-backend.railway.app/api/llm-models
  ```
  Expected: < 2 seconds

- [ ] **Frontend Performance**
  - [ ] Dashboard loads in < 3 seconds
  - [ ] Equity curves render smoothly
  - [ ] No lag when switching tabs

- [ ] **WebSocket Performance**
  - [ ] Real-time updates arrive within 5 seconds
  - [ ] No connection drops
  - [ ] Reconnects automatically if disconnected

---

## Battle Royale Verification

If using multiple LLM models:

- [ ] **All Models Configured**
  - [ ] GPT-4 API keys set
  - [ ] Claude API keys set
  - [ ] Gemini API keys set
  - [ ] DeepSeek API keys set (optional)
  - [ ] Qwen API keys set (optional)
  - [ ] Grok API keys set (optional)

- [ ] **All Models Active**
  - [ ] `/api/llm-models` shows all models
  - [ ] Each model shows "connected: true"
  - [ ] Leaderboard displays all models

- [ ] **Trading Activity**
  - [ ] Models making trade decisions
  - [ ] Positions appearing in dashboard
  - [ ] Equity curves updating

---

## Cost Monitoring Checklist

- [ ] **Railway Usage**
  - [ ] Review usage dashboard weekly
  - [ ] Estimated cost within budget
  - [ ] No unexpected resource spikes

- [ ] **LLM API Costs**
  - [ ] Monitor OpenAI/Anthropic usage
  - [ ] Set spending limits if available
  - [ ] Track cost per decision/trade

---

## Final Verification

Run all these commands and verify they work:

```bash
# Backend health
curl https://your-backend.railway.app/

# LLM models
curl https://your-backend.railway.app/api/llm-models

# Leaderboard
curl https://your-backend.railway.app/api/battle/alpaca-leaderboard

# Equity curves
curl https://your-backend.railway.app/api/alpaca-equity-curves

# Frontend
open https://your-frontend.railway.app
```

**All tests passing?** ✅ **Your deployment is complete and verified!**

---

## Common Issues & Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| CORS error | Update `CORS_ORIGINS` in backend, redeploy |
| WebSocket fails | Check `VITE_WS_URL` uses `wss://` protocol |
| No data in dashboard | Verify `VITE_API_URL` is correct |
| Cron not running | Check volume is attached, environment variables set |
| Database errors | Verify `DATABASE_PATH=/data` and volume mounted |
| 500 errors | Check backend logs for Python errors |

---

## Support

If you encounter issues:

1. Check Railway service logs
2. Verify all environment variables
3. Review DEPLOYMENT.md for detailed steps
4. Open GitHub issue with error logs

**Happy Trading!** 🚀📈
