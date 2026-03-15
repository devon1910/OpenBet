# OpenBet — Football Prediction Engine

A professional-grade football match prediction system combining statistical modelling, machine learning, and Claude AI reasoning to surface high-confidence betting opportunities across Europe's top leagues.

---

## How It Works

Predictions are generated through a three-layer stacking ensemble:

```
Layer 1 — Poisson Model        (attack/defence strength, home advantage)
Layer 2 — XGBoost Classifier   (ELO ratings, form, xG, head-to-head, odds)
Layer 3 — Meta-Learner         (blends Layer 1 + 2 with bookmaker implied probs)
           └── Claude AI        (contextual overlay: motivation, injuries, rivalries)
```

Each pick is filtered by a minimum **value edge** — the gap between the model's probability and the bookmaker's implied probability — ensuring only bets where the model has an edge are surfaced.

---

## Features

- **9 competitions** — Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Eredivisie, Primeira Liga, Championship, Champions League
- **ELO rating system** — continuously updated after every match result
- **xG enrichment** — expected goals data fed into features
- **Live odds integration** — bookmaker odds from The Odds API, averaged across multiple bookmakers and normalised to remove overround
- **Claude AI reasoning layer** — flags title deciders, relegation battles, derbies, fixture congestion, manager bounce, and key injuries
- **Automated 6-hour pipeline** — data sync → odds → features → predictions, fully automatic
- **Self-improving model** — weekly performance evaluation with automatic retraining when accuracy drops
- **Backtesting** — walk-forward backtest with ROI and per-matchday breakdown
- **Admin panel** — protected by JWT authentication, visible only after login

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI (Python 3.11) |
| Database | PostgreSQL (async via asyncpg + SQLAlchemy 2.0) |
| ML | XGBoost, scikit-learn, SciPy (Poisson) |
| AI reasoning | Claude claude-sonnet-4-20250514 (Anthropic API) |
| Scheduler | APScheduler (in-process, no Redis required) |
| Auth | JWT (python-jose) + bcrypt |
| Frontend | Vanilla HTML/CSS/JS (single file, no build step) |

---

## Prerequisites

- Python 3.11+
- PostgreSQL 15+ (local or Neon)
- API keys (see [Configuration](#configuration))

---

## Local Setup

**1. Clone and create virtual environment**
```bash
git clone <repo-url>
cd OpenBet
python -m venv .venv
```

**2. Activate and install dependencies**

Windows:
```powershell
.venv\Scripts\activate
pip install -e .
```

macOS/Linux:
```bash
source .venv/bin/activate
pip install -e .
```

**3. Configure environment**

Copy `.env.example` to `.env` and fill in your values:
```bash
cp .env.example .env
```

**4. Set up the database**

Start Postgres (or point `DATABASE_URL` at Neon/another managed instance), then run migrations:
```bash
alembic upgrade head
```

**5. Seed historical data**

Pulls 5 seasons of match data across all supported leagues:
```bash
python scripts/seed_historical.py
```
This takes ~10–15 minutes on a free API tier due to rate limiting.

**6. Train the initial model**
```bash
python scripts/train_initial.py
```

**7. Start the server**
```bash
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## Configuration

All settings are loaded from `.env`. Required values:

```env
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/openbet
DATABASE_URL_SYNC=postgresql://user:pass@host:5432/openbet

# API Keys
FOOTBALL_DATA_API_KEY=   # football-data.org — match data (free tier)
API_FOOTBALL_KEY=        # api-sports.io — xG and injury enrichment (free tier)
ANTHROPIC_API_KEY=       # Anthropic — Claude reasoning layer
ODDS_API_KEY=            # the-odds-api.com — bookmaker odds

# Admin auth
ADMIN_USERNAME=admin
ADMIN_PASSWORD_HASH=     # generate with: python scripts/generate_password_hash.py
JWT_SECRET=              # long random string, keep secret
```

Optional tuning:
```env
STRAIGHT_WIN_THRESHOLD=0.55    # minimum probability to recommend a win pick
DOUBLE_CHANCE_THRESHOLD=0.75   # minimum probability for a double chance pick
MAX_PICKS_PER_MATCHDAY=9       # cap on daily picks
MIN_VALUE_EDGE=0.05            # minimum edge over bookmaker implied prob
CLAUDE_MAX_ADJUSTMENT=0.10     # max Claude confidence adjustment (±)
ELO_K_FACTOR=32.0
```

---

## Admin Panel

Click **Admin** in the top-right corner. You will be prompted to log in with your admin credentials.

Once authenticated, the pipeline controls are available:

| Button | Action |
|--------|--------|
| Sync Match Data | Pulls latest results and fixtures from football-data.org |
| Fetch Odds | Updates bookmaker odds for upcoming matches |
| Build Features | Recalculates ELO, form, xG, and all ML features |
| Train Model | Retrains XGBoost + meta-learner on current data |
| Resolve Outcomes | Marks picks as WIN/LOSS based on finished results |
| Run Backtest | Walk-forward backtest over the last 8 weeks |
| Refresh Status | Reloads DB and model status stats |

The **Logout** button appears inside the panel. The JWT session expires after 8 hours.

---

## Automated Pipeline

The scheduler runs automatically when the server starts — no separate process needed.

| Schedule | Job |
|----------|-----|
| Every 6 hours (00:00, 06:00, 12:00, 18:00 UTC) | Sync → Odds → Features → Predictions |
| Daily 23:30 UTC | Resolve pick outcomes |
| Every Monday 04:00 UTC | Evaluate model performance, retrain if needed |

A public **Engine status** footer at the bottom of every page shows the last run time and result for each job.

---

## Deployment (Azure + Neon + Vercel)

### Database — Neon
1. Create a project at [neon.tech](https://neon.tech)
2. Copy the connection string into `DATABASE_URL` / `DATABASE_URL_SYNC` (append `?sslmode=require`)
3. Run `alembic upgrade head` against the Neon connection

### API — Azure App Service
1. Create a Linux App Service (Python 3.11)
2. Set startup command:
   ```
   uvicorn src.api.app:app --host 0.0.0.0 --port 8000
   ```
3. Add all `.env` values as **Application Settings** in the Azure portal
4. The APScheduler runs in-process — no separate worker or Redis instance required

### Frontend — Vercel
1. The frontend is `static/index.html` — a single self-contained file
2. Update the `fetch()` base URLs to point at your Azure App Service domain (or use a `/api` proxy rewrite rule in `vercel.json`)
3. Deploy via Vercel CLI or connect the GitHub repo

---

## Project Structure

```
OpenBet/
├── src/
│   ├── api/
│   │   ├── auth.py                 # JWT utilities
│   │   ├── app.py                  # FastAPI app factory + scheduler wiring
│   │   └── routes/
│   │       ├── admin.py            # Admin endpoints (JWT protected)
│   │       ├── auth_routes.py      # POST /auth/login
│   │       ├── picks.py            # GET /picks/date/{date}
│   │       ├── predictions.py      # GET /predictions/matchday/...
│   │       ├── performance.py      # GET /performance/...
│   │       ├── health.py           # GET /health, GET /pipeline-status
│   │       └── teams.py
│   ├── collectors/
│   │   ├── football_data.py        # football-data.org client
│   │   ├── api_football.py         # api-sports.io client (xG, injuries)
│   │   └── odds_api.py             # the-odds-api.com client
│   ├── features/
│   │   ├── builder.py              # Assembles feature vectors per match
│   │   ├── elo.py                  # ELO rating calculation
│   │   ├── form.py                 # Recent form features
│   │   ├── xg.py                   # Expected goals features
│   │   └── strength.py             # Attack/defence strength
│   ├── models_ml/
│   │   ├── poisson.py              # Dixon-Coles Poisson model
│   │   ├── xgboost_model.py        # XGBoost classifier
│   │   ├── ensemble.py             # Stacking ensemble + meta-learner
│   │   └── training.py             # Training pipeline
│   ├── engine/
│   │   ├── betting.py              # Value edge filtering and pick selection
│   │   └── claude_reasoning.py     # Claude AI contextual analysis
│   ├── learning/
│   │   ├── tracker.py              # Outcome resolution (WIN/LOSS)
│   │   ├── retrainer.py            # Auto-retraining logic
│   │   └── backtester.py           # Walk-forward backtest
│   └── workers/
│       └── scheduler.py            # APScheduler jobs + pipeline status
├── scripts/
│   ├── seed_historical.py          # One-time historical data backfill
│   ├── train_initial.py            # One-time initial model training
│   └── generate_password_hash.py  # Generate bcrypt hash for admin password
├── static/
│   └── index.html                  # Frontend (single file)
├── trained_models/                 # Saved model artefacts (gitignored)
├── pyproject.toml
└── .env
```

---

## API Reference

All endpoints return JSON.

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/` | — | Serves the frontend |
| `GET` | `/health` | — | Health check |
| `GET` | `/pipeline-status` | — | Last pipeline run times and status |
| `GET` | `/picks/date/{YYYY-MM-DD}` | — | Picks for a date (runs engine if none cached) |
| `GET` | `/picks/today` | — | Shorthand for today's picks |
| `GET` | `/picks/history` | — | Historical picks with outcomes |
| `POST` | `/auth/login` | — | Returns JWT token |
| `POST` | `/admin/sync-data` | JWT | Sync match data |
| `POST` | `/admin/fetch-odds` | JWT | Fetch bookmaker odds |
| `POST` | `/admin/build-features` | JWT | Build/rebuild features |
| `POST` | `/admin/train` | JWT | Train model |
| `POST` | `/admin/resolve-outcomes` | JWT | Resolve pick outcomes |
| `POST` | `/admin/backtest` | JWT | Run backtest |
| `GET` | `/admin/status` | JWT | System stats |

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Supported Competitions

| Code | Competition | Country |
|------|-------------|---------|
| PL | Premier League | England |
| ELC | Championship | England |
| PD | La Liga | Spain |
| BL1 | Bundesliga | Germany |
| SA | Serie A | Italy |
| FL1 | Ligue 1 | France |
| DED | Eredivisie | Netherlands |
| PPL | Primeira Liga | Portugal |
| CL | Champions League | Europe |

---

## Licence

Private — all rights reserved.
