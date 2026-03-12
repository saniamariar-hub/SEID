# SEID Engine — Social Engineering & Intrusion Detection System

---

## Project Overview

SEID Engine is a machine learning–based system designed to detect **social engineering attacks**, specifically **phishing emails** and **smishing (SMS-based phishing)** messages.

The system combines traditional natural language processing techniques with transformer-based deep learning models to analyze textual patterns and contextual semantics in digital communications.

The system enables users to analyze messages in real time and determine the likelihood that a message is malicious.

---

## System Architecture

The SEID Engine follows a layered architecture consisting of three major components.

### Frontend Layer

A web-based dashboard built using modern frontend technologies allows users to submit messages for analysis and view prediction results.

**Technologies Used**

- React 18  
- Vite  
- Tailwind CSS  
- Axios  

The dashboard provides a simple and responsive interface for interacting with the detection engine.

### Backend Layer

The backend server manages API communication between the frontend and the machine learning models.

**Technologies Used**

- Python  
- FastAPI  
- Uvicorn  

The backend performs preprocessing, model inference, ensemble scoring, and risk classification before returning results to the user.

### Machine Learning Layer

The machine learning layer loads trained models and processes incoming text messages to determine whether they are malicious.

It performs the following operations:

- Text preprocessing  
- Feature extraction  
- Model inference  
- Ensemble prediction  
- Risk classification  

---

## Machine Learning Models

The SEID Engine uses an **ensemble machine learning architecture** combining traditional NLP models with transformer-based deep learning.

### TF-IDF + Logistic Regression

This model converts text into numerical features using **TF-IDF vectorization** and classifies messages using **Logistic Regression**.

**Implementation Tools**

- Scikit-learn  
- TF-IDF Vectorizer  
- Logistic Regression  
- Joblib (model serialization)

This model captures statistical word frequency patterns commonly found in phishing messages.

---

### RoBERTa Transformer Model

RoBERTa is a transformer-based deep learning model that understands contextual relationships within text.

**Implementation Tools**

- HuggingFace Transformers  
- PyTorch  
- RoBERTa-base architecture  

This model helps detect subtle linguistic patterns and contextual manipulation techniques used in social engineering attacks.

---

## Ensemble Prediction

Predictions from both models are combined using a weighted scoring mechanism.

```
final_score = 0.6 × roberta_score + 0.4 × tfidf_score
```

This hybrid approach improves detection reliability by combining contextual understanding with statistical analysis.

---

## Risk Classification

Prediction probabilities are mapped into predefined risk tiers.

| Probability | Risk Tier |
|-------------|-----------|
| < 0.2 | Low |
| 0.2 – 0.5 | Medium |
| 0.5 – 0.8 | High |
| > 0.8 | Critical |

These risk tiers help users quickly interpret the severity of potential threats.

---

## Security Modes

The SEID Engine provides configurable detection modes.

| Mode | Description |
|-----|-------------|
| Balanced | General detection performance |
| High Recall | Reduces missed attacks |
| Low False Positive | Reduces false alarms |

These modes allow the system to adjust sensitivity based on operational requirements.

---

## Dataset Sources

The training dataset was constructed by combining multiple publicly available datasets.

### Enron Email Dataset

Used as a source of legitimate corporate email communications.

### SMS Spam Collection Dataset

Contains SMS messages labeled as spam or legitimate.

### Phishing Email Corpora

Datasets containing real phishing campaign messages.

All datasets were converted into a **unified canonical format** before training.

---

## Data Processing Pipeline

Custom preprocessing scripts were used to standardize and merge the datasets.

Scripts used in dataset preparation:

```
process_enron_canonical.py
process_sms_canonical.py
process_phishing_email_canonical.py
build_master_corpus_v2.py
```

These scripts generate the **master training corpus** used for machine learning model training.

---

## API Endpoints

The backend exposes REST API endpoints used by the frontend dashboard.

### Health Check

```
GET /health
```

Returns the system status and model availability.

---

### Message Prediction

```
POST /predict
```

Example request:

```json
{
  "text": "Your account has been suspended. Click here to verify.",
  "channel": "email",
  "mode": "balanced"
}
```

Example response:

```json
{
  "probability": 0.998,
  "risk_tier": "Critical",
  "is_malicious": true
}
```

---

## Technical Stack

### Backend

- Python  
- FastAPI  
- Uvicorn  
- Scikit-learn  
- PyTorch  
- Transformers  
- Pandas  
- NumPy  
- Joblib  

### Frontend

- React  
- Vite  
- Tailwind CSS  
- Axios  

### Machine Learning

- TF-IDF feature extraction  
- Logistic Regression  
- Transformer-based NLP (RoBERTa)

---

## Running the Project

### Backend Setup

Install dependencies:

```
pip install -r requirements.txt
```

Start the FastAPI server:

```
uvicorn app:app --reload
```

---

### Frontend Setup

Navigate to the frontend directory and install dependencies:

```
npm install
```

Start the development server:

```
npm run dev
```

The dashboard will be available at:

```
http://localhost:5173
```

---

## Project Structure

```
SEID_ENGINE
│
├── models
│   ├── tfidf_model
│   └── roberta_malicious_classifier
│
├── engine
│   ├── preprocessing.py
│   ├── inference.py
│   ├── ensemble.py
│   └── risk_tiers.py
│
├── frontend
│   └── React + Vite Dashboard
│
├── app.py
├── seid_engine.py
└── requirements.txt
```

---

## Key Features

- Cross-channel phishing and smishing detection  
- Ensemble machine learning architecture  
- Transformer-based language understanding  
- Real-time message analysis  
- Configurable detection modes  
- Interactive web dashboard  
- Modular and scalable architecture  

---

## Future Work

Planned improvements include:

- Integration with additional communication platforms  
- Real-time monitoring and alerting systems  
- Advanced analytics and visualization features  
- Continuous model retraining pipelines  

---

## License

This project is developed for **academic and research purposes**.
