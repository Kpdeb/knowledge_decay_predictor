# 🧠 AI Knowledge Decay Predictor

Predicts when you'll forget concepts using the **Ebbinghaus Forgetting Curve + Spaced Repetition + ML models**, then schedules optimal revision sessions.

---

## 🏗️ Architecture

```
Frontend (React)  →  Backend (FastAPI)  →  PostgreSQL
                          ↓
                   ML Service (Python/sklearn)
```

---

## 📂 Folder Structure

```
knowledge-decay-predictor/
├── frontend/          # React dashboard
├── backend/           # FastAPI REST API
├── ml-service/        # Python ML prediction service
├── database/          # SQL schema
├── docs/              # Architecture & API docs
├── docker-compose.yml
└── .env.example
```

---

## 🚀 Quick Start

### Option 1 — Docker (recommended)

```bash
cp .env.example .env
docker-compose up --build
```

- Frontend: http://localhost:3000  
- Backend API: http://localhost:8000  
- ML Service: http://localhost:8001  
- API Docs: http://localhost:8000/docs  

### Option 2 — Manual

**1. Database**
```bash
psql -U postgres -c "CREATE DATABASE knowdecay;"
psql -U postgres -d knowdecay -f database/schema.sql
```

**2. Backend**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**3. ML Service**
```bash
cd ml-service
pip install -r requirements.txt
python training/train.py          # train model first
uvicorn main:app --reload --port 8001
```

**4. Frontend**
```bash
cd frontend
npm install
npm start
```

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/topics` | Add a topic |
| GET | `/topics` | List all topics |
| POST | `/quiz` | Submit quiz result |
| GET | `/prediction/{topic_id}` | Get retention prediction |
| POST | `/schedule` | Generate revision schedule |
| GET | `/users/{user_id}/dashboard` | Dashboard summary |

---

## 🧪 ML Model

Two approaches:
1. **Rule-based**: Ebbinghaus forgetting curve `R = e^(-t/S)`
2. **ML-based**: Random Forest trained on synthetic + real study data

Features: `time_since_last_review`, `quiz_score`, `difficulty`, `review_count`, `study_duration`  
Output: retention probability (0–1)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, Recharts, Axios |
| Backend | FastAPI, SQLAlchemy, Pydantic |
| Database | PostgreSQL 15 |
| ML | scikit-learn, pandas, joblib |
| Auth | JWT (python-jose) |
| Containers | Docker + Docker Compose |

---

## 📊 Spaced Repetition Schedule

| Review # | Interval |
|----------|----------|
| 1st | 1 day |
| 2nd | 3 days |
| 3rd | 7 days |
| 4th | 14 days |
| 5th+ | 30 days |

---

## 👥 Demo Credentials

```
Email: demo@knowdecay.ai
Password: demo1234
```
