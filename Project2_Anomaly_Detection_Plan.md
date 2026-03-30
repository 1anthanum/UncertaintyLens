# Project 2: Real-Time Anomaly Detection Dashboard

## Project Overview

An end-to-end ML system that simulates streaming data (server metrics, financial data, or IoT sensors), detects anomalies in real-time using deep learning, and displays results on a live dashboard with alerting.

**Key Differentiator from Project 1**: Project 1 shows model expertise (architectures, explainability). This project shows **systems engineering** — the full pipeline from data ingestion to deployed, monitored inference.

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Data Source  │────▶│  ML Pipeline │────▶│  Alert Engine│────▶│  Dashboard   │
│  (Simulator)  │     │  (Inference)  │     │  (Rules)     │     │  (Streamlit) │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
│ Generates          │ LSTM/Transformer   │ Threshold +        │ Real-time
│ realistic time     │ autoencoder for    │ anomaly scoring    │ charts, alerts,
│ series with        │ anomaly scoring    │ + notification     │ model metrics
│ injected anomalies │                    │                    │
```

---

## What You'll Build

### 1. Data Simulator (`data/simulator.py`)
Generates realistic streaming time series with controllable anomaly injection:
- **Normal patterns**: seasonal trends, daily/weekly cycles, noise
- **Anomaly types**: point anomalies (spikes), contextual anomalies (wrong time), collective anomalies (sustained shift)
- **Domains**: Server CPU/memory metrics, stock price, or sensor readings (pick one)

### 2. Anomaly Detection Models (`models/`)
Two complementary approaches:
- **LSTM Autoencoder**: Learns to reconstruct normal patterns; high reconstruction error = anomaly
- **Transformer-based detector**: Attention-based model for capturing long-range dependencies
- Both output an **anomaly score** (0-1) for each time step

### 3. Real-Time Dashboard (`app.py`)
Streamlit app with:
- **Live time series chart** that updates with simulated streaming data (Plotly)
- **Anomaly overlay**: detected anomalies highlighted in red on the chart
- **Anomaly score gauge**: real-time anomaly probability meter
- **Alert log**: timestamped list of detected anomalies with severity
- **Model selector**: switch between LSTM and Transformer detector
- **Controls**: adjust sensitivity threshold, data speed, anomaly injection rate

### 4. System Metrics Panel
- Model inference latency (ms per prediction)
- Detection accuracy (precision, recall, F1 on injected anomalies)
- False positive rate
- Rolling performance over time

---

## Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| Models | PyTorch (LSTM, Transformer) | Your core expertise |
| App | Streamlit | Fast prototyping, live updates |
| Charts | Plotly | Interactive, real-time capable |
| Data | NumPy + pandas | Time series generation |
| Containerization | Docker + docker-compose | Shows deployment skills |
| CI | GitHub Actions | Auto-test on push |
| Deployment | Hugging Face Spaces | Free, GPU-capable hosting |

---

## Repository Structure

```
realtime-anomaly-detector/
├── README.md                          # Hero README (see template below)
├── app.py                             # Streamlit main app
├── Dockerfile                         # Container for deployment
├── docker-compose.yml                 # Multi-service setup
├── requirements.txt
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions: lint + test
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── simulator.py               # Streaming data generator
│   │   └── preprocessor.py            # Windowing, normalization
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_autoencoder.py        # LSTM Autoencoder
│   │   ├── transformer_detector.py    # Transformer-based detector
│   │   └── base.py                    # Abstract base class
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── scoring.py                 # Anomaly scoring logic
│   │   └── alerts.py                  # Alert rules and thresholds
│   └── visualization/
│       ├── __init__.py
│       └── dashboard.py               # Plotly chart components
├── weights/
│   ├── lstm_autoencoder.pt
│   └── transformer_detector.pt
├── notebooks/
│   └── training.ipynb                 # Model training notebook
├── tests/
│   ├── test_simulator.py
│   ├── test_models.py
│   └── test_scoring.py
└── configs/
    └── default.yaml                   # Configurable parameters
```

---

## 3-Week Build Plan

### Week 1: Core Engine (10-12 hrs)

**Day 1-2: Data Simulator (4 hrs)**
- Build `simulator.py` with configurable patterns
- Implement 3 anomaly types (point, contextual, collective)
- Unit test: verify anomaly injection works correctly

**Day 3-4: Models (4 hrs)**
- Port/adapt your existing LSTM code into autoencoder architecture
- Implement reconstruction error based anomaly scoring
- Pre-train on simulated normal data, save weights

**Day 5: Scoring Pipeline (2-3 hrs)**
- Sliding window inference
- Anomaly score calculation (reconstruction error → normalized score)
- Threshold-based alert triggering

**Deliverable**: Working detection pipeline that can process a batch of time series data and output anomaly scores.

### Week 2: Dashboard + Transformer (10-12 hrs)

**Day 1-3: Streamlit Dashboard (6 hrs)**
- Live-updating Plotly chart with simulated stream
- Anomaly highlighting overlay
- Controls panel (threshold slider, speed, model selector)
- Alert log component

**Day 4-5: Transformer Model (4 hrs)**
- Implement Transformer-based detector as alternative
- Add model switching in dashboard
- Performance comparison panel

**Deliverable**: Working interactive dashboard with two model options.

### Week 3: Polish + Deploy (8-10 hrs)

**Day 1-2: Docker + CI (3 hrs)**
- Dockerfile for containerized deployment
- docker-compose.yml
- GitHub Actions CI pipeline (lint + tests)

**Day 3: README + Documentation (2 hrs)**
- Hero README with GIF demo, architecture diagram
- Screenshots of dashboard
- Performance benchmarks

**Day 4-5: Deploy + Test (3 hrs)**
- Deploy to Hugging Face Spaces
- End-to-end testing
- Final polish

---

## README Template

```markdown
# 🔍 Real-Time Anomaly Detection Dashboard

[![Live Demo](https://img.shields.io/badge/Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/YOUR_SPACE)
[![CI](https://github.com/1anthanum/realtime-anomaly-detector/actions/workflows/ci.yml/badge.svg)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](Dockerfile)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)]()

An end-to-end ML system for real-time anomaly detection in streaming time series data, featuring LSTM Autoencoder and Transformer-based detectors with an interactive monitoring dashboard.

![Dashboard Demo](assets/demo.gif)

## 🏗️ Architecture

[Insert architecture diagram]

## ✨ Features

- **Real-time streaming visualization** with Plotly live charts
- **Dual detection models**: LSTM Autoencoder + Transformer detector
- **Interactive controls**: adjustable sensitivity, anomaly injection, data speed
- **Alert system**: timestamped anomaly log with severity classification
- **Performance monitoring**: inference latency, precision/recall, false positive rate
- **Containerized**: Docker-ready for production deployment

## 🚀 Quick Start

### Local
pip install -r requirements.txt
streamlit run app.py

### Docker
docker-compose up

### Live Demo
👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_SPACE)

## 📊 Model Performance

| Model | Precision | Recall | F1 | Latency (ms) |
|-------|-----------|--------|-----|---------------|
| LSTM Autoencoder | 0.94 | 0.91 | 0.92 | 12 |
| Transformer | 0.96 | 0.89 | 0.92 | 18 |

## 🛠️ Tech Stack

PyTorch · Streamlit · Plotly · Docker · GitHub Actions

## 📄 Related Work

- [Time Series Classification Explorer](link) — My other project on multi-architecture time series analysis
- [IEEE Paper: Physics-Informed Neural Networks](link) — Published research

## 📝 License

MIT
```

---

## What Makes This Project Stand Out for Recruiters

1. **End-to-end system**: Not just a model — it's a complete pipeline (data → inference → UI → alerting)
2. **Docker + CI**: Shows production engineering mindset, not just notebook prototyping
3. **Real-time**: Much more impressive than batch processing demos
4. **Two model comparison**: Shows architectural knowledge and ability to evaluate tradeoffs
5. **Interactive dashboard**: Recruiters can actually *play with it*
6. **Tests**: Having a `tests/` directory with actual tests immediately signals professionalism
7. **Clean code structure**: Modular, well-organized, shows software engineering discipline

---

## Honest Assessment: Risks & Mitigations

**Risk 1**: Domain overlap with Project 1 (both time series)
- **Mitigation**: Project 1 = model expertise. Project 2 = systems engineering. Frame them as complementary, not redundant. In interviews, explain: "One shows my research depth, the other shows I can build production systems."

**Risk 2**: "Real-time" in Streamlit is simulated, not true streaming
- **Mitigation**: This is fine for a portfolio project. But be honest about it — call it "simulated streaming" in README, and mention that a production system would use Kafka/Flink. This shows you understand the gap between demo and production.

**Risk 3**: 20-25 hours might be tight
- **Mitigation**: Prioritize ruthlessly. If time runs short, cut the Transformer model and focus on making the LSTM + dashboard bulletproof. One polished model > two half-baked models.
