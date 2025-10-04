# 🔑 API Keys Guide - Where to Get Everything

## ✅ **PRIORITY ORDER** (Get these in order)

---

## 🟢 **REQUIRED (Get First!)**

### 1. Google Gemini API ⭐ **MOST IMPORTANT**
**Cost**: **FREE** (60 requests/minute)
**Why**: Powers NLU (Natural Language Understanding) and text generation
**Time**: 2 minutes

**How to Get**:
1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click **"Get API key"** or **"Create API key"**
4. Copy the key (starts with `AIza...`)
5. Paste in `.env`: `GEMINI_API_KEY=AIza...`

**Test it**:
```powershell
# Add to backend/.env, then:
curl http://localhost:8000/health
# Should show "gemini": "configured"
```

---

## 🟡 **RECOMMENDED (For Demo)**

### 2. Mapbox Token (For Map Visualization)
**Cost**: **FREE** (50,000 map loads/month)
**Why**: Powers the interactive traffic map
**Time**: 3 minutes

**How to Get**:
1. Go to: https://account.mapbox.com/auth/signup/
2. Sign up (free account)
3. Go to **Account** → **Access Tokens**
4. Copy your **Default public token** (starts with `pk.`)
5. Add to `frontend/src/components/TrafficMap.js`:
   ```javascript
   const MAPBOX_TOKEN = 'pk.your_token_here';
   ```

**Why FREE is OK**: Demo uses limited map interactions

---

### 3. Auth0 (For User Authentication)
**Cost**: **FREE** (7,000 active users/month)
**Why**: Secure user login and JWT tokens
**Time**: 5 minutes

**How to Get**:
1. Go to: https://auth0.com/signup
2. Sign up (free account)
3. Create a **"Regular Web Application"**
4. Go to **Settings** tab:
   - Copy **Domain** (e.g., `dev-abc123.us.auth0.com`)
   - Copy **Client ID**
   - Copy **Client Secret**
5. Add to `backend/.env`:
   ```env
   AUTH0_DOMAIN=dev-abc123.us.auth0.com
   AUTH0_CLIENT_ID=your_client_id
   AUTH0_CLIENT_SECRET=your_client_secret
   ```
6. Set **Allowed Callback URLs**: `http://localhost:3000/callback`
7. Set **Allowed Logout URLs**: `http://localhost:3000`

**For Now**: Use `SKIP_AUTH_VERIFICATION=true` to bypass

---

## 🟠 **OPTIONAL (For Full Features)**

### 4. ElevenLabs (For Voice Narration)
**Cost**: **FREE** (10,000 characters/month = ~20 minutes of audio)
**Why**: Converts simulation summaries to audio
**Time**: 3 minutes

**How to Get**:
1. Go to: https://elevenlabs.io/sign-up
2. Sign up (free account)
3. Go to **Profile** → Copy **API Key**
4. Go to **Voice Library** → Pick a voice (e.g., "Adam")
5. Copy **Voice ID**
6. Add to `backend/.env`:
   ```env
   ELEVENLABS_API_KEY=your_key_here
   ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
   ```

**Test Voice**:
- Voice IDs: Adam=`21m00Tcm4TlvDq8ikWAM`, Rachel=`21m00Tcm4TlvDq8ikWsZ`

---

### 5. Agentuity (For Agent Orchestration)
**Cost**: Check their website (may have free tier)
**Why**: Orchestrates the 3-agent workflow
**Time**: 5 minutes

**How to Get**:
1. Go to: https://www.agentuity.com/
2. Sign up for an account
3. Go to **Settings** → **API Keys**
4. Generate a new API key
5. Copy webhook URL from dashboard
6. Add to `backend/.env`:
   ```env
   AGENTUITY_API_KEY=your_key_here
   AGENTUITY_WEBHOOK_URL=https://your-endpoint.com/webhook
   ```

**For Now**: Backend works in mock mode without Agentuity

---

## 🔵 **NOT NEEDED (Already Working)**

### ✅ MATLAB
**Status**: Running in **mock mode**
**Why**: Your simulation works without MATLAB license
**Action**: No setup needed!

### ✅ Machine Learning Models
**Status**: Already **trained and loaded**
**Location**: `ml/models/` (4 files)
**Action**: No setup needed!

---

## 📋 **QUICK SETUP CHECKLIST**

### Minimum Viable Demo (5 minutes)
- [x] ✅ ML Models trained
- [ ] 🟢 Get Gemini API key
- [ ] 🟡 Get Mapbox token (optional but nice)
- [ ] Update `backend/.env`
- [ ] Restart backend

### Full Demo (15 minutes)
- [ ] All above +
- [ ] 🟡 Auth0 setup
- [ ] 🟠 ElevenLabs key
- [ ] Test all endpoints

### Competition Ready (30 minutes)
- [ ] All above +
- [ ] 🟠 Agentuity setup
- [ ] Deploy agents
- [ ] Record demo video

---

## 🚀 **IMMEDIATE NEXT STEPS**

### Step 1: Get Gemini API Key (2 min)
```
1. Go to: https://makersuite.google.com/app/apikey
2. Get API key
3. Add to backend/.env
```

### Step 2: Update .env File
```powershell
notepad backend\.env
# Paste your Gemini key
# Save and close
```

### Step 3: Restart Backend
```powershell
# Close backend terminal (Ctrl+C)
# Restart:
python backend/main.py
```

### Step 4: Test It!
```
http://localhost:8000/health
# Should show all services configured
```

---

## 💰 **COST SUMMARY**

| Service | Free Tier | Needed For |
|---------|-----------|------------|
| **Gemini API** | 60 req/min | ✅ **CRITICAL** - NLU |
| **Mapbox** | 50K loads/mo | 🟡 Map visualization |
| **Auth0** | 7K users/mo | 🟡 Authentication |
| **ElevenLabs** | 10K chars/mo | 🟠 Voice narration |
| **Agentuity** | TBD | 🟠 Agent orchestration |

**Total Monthly Cost for Demo**: **$0** (All free tiers!)

---

## 🎯 **FOR HACKATHON DEMO**

### Minimum (Works Right Now!)
- ✅ ML Models (trained)
- ✅ Backend API (running)
- ✅ Frontend UI (running)
- ⚠️ Gemini API (needs key)

### Recommended (Impressive Demo)
- All above +
- Mapbox (better map)
- Auth0 (secure login)

### Maximum (Win All Prizes)
- All above +
- ElevenLabs (audio)
- Agentuity (agent orchestration)

---

## 📞 **HELP**

### API Key Not Working?
1. Check for typos
2. Verify key is active in dashboard
3. Check rate limits
4. See error in backend terminal

### Skip Auth for Now?
```env
# In backend/.env
SKIP_AUTH_VERIFICATION=true
```

### Test Backend Without Keys?
```
# Mock mode works for:
- MATLAB (simulation)
- OSM Data (road networks)
- ML Models (if trained)
```

---

## 🎉 **READY TO GO!**

**With just Gemini API key, you have:**
- ✅ Full ML predictions
- ✅ NLU prompt parsing
- ✅ Text summarization
- ✅ Interactive frontend
- ✅ All core features working

**Get that Gemini key and you're competition-ready!** 🚀

---

**Priority**: Get Gemini API key first (2 minutes), everything else is optional for demo!
