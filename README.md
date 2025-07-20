# Heart Disease Prediction - End-to-End ML Application

Welcome! This project is a complete machine learning solution for predicting heart disease, featuring data processing, model training, evaluation, and a secure, production-ready API.

---

##  What Does This Project Do?

- **Loads and preprocesses heart disease data**
- **Trains a robust ML pipeline** (with scaling and logistic regression)
- **Evaluates the model** and saves a confusion matrix
- **Serves predictions via a FastAPI REST API**
- **Secures the API with OAuth2 Bearer Token authentication**

---

## 🗂️ Project Structure

- `src/data_injection.py` — Loads and splits the data
- `src/model_training.py` — Trains and evaluates the ML pipeline
- `src/main.py` — FastAPI app for predictions (with authentication)
- `model/pipeline.pkl` — The trained pipeline (preprocessing + model)
- `data/` — Raw and processed data
- `reports/` — Evaluation reports and confusion matrix
- `config/params.yaml` — Configuration parameters

---

## 🛠️ Getting Started

### 1. **Install dependencies**

```bash
pip install -r requirements.txt
```

### 2. **Prepare the data**

```bash
python src/data_injection.py
```

### 3. **Train the model**

```bash
python src/model_training.py
```

### 4. **Start the API server**

```bash
uvicorn src.main:app --reload
```

---

## Authentication (OAuth2 Bearer Token)

All prediction requests require authentication!

### 1. **Get a token**

Send a POST request to `/token` with the following credentials:

- **Username:** `user`
- **Password:** `password`

Example using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/token" -d "username=user&password=password"
```

This will return a token like:

```json
{ "access_token": "supersecrettoken", "token_type": "bearer" }
```

### 2. **Make a prediction**

Use the token in the `Authorization` header:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Authorization: Bearer supersecrettoken" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }'
```

---

## API Reference

- **POST `/token`** — Get an access token (OAuth2)
- **POST `/predict`** — Get a heart disease prediction (requires Bearer token)
- **GET `/`** — Welcome message

You can also explore and test the API interactively at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (Swagger UI).
