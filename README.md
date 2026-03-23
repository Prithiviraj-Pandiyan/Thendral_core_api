# Thendral Core API

Phase 1 of **Thendral** — the foundational backend and ML-serving layer for a long-term, modular AI system.

---

## Technical Architecture Summary

`Thendral Core API` is the Phase 1 backend for a modular AI platform. In this phase, the system is designed as a layered service that accepts input through a FastAPI interface, runs shared preprocessing, sends the prepared data into an ML inference pipeline, and returns structured responses. The architecture is intentionally split into reusable modules so future phases can add more models, pipelines, and intelligence services without restructuring the entire codebase.

### High-Level System Flow

```text
Client / Consumer
       |
       v
+----------------------+
| FastAPI API Layer    |
| - request handling   |
| - validation         |
| - response shaping   |
+----------------------+
       |
       v
+----------------------+
| Preprocessing Layer  |
| - text cleaning      |
| - normalization      |
| - shared transforms  |
+----------------------+
       |
       v
+----------------------+
| Inference Layer      |
| - load artifacts     |
| - run prediction     |
| - map output         |
+----------------------+
       |
       v
+----------------------+
| Structured API       |
| Response             |
+----------------------+
```

### Training and Serving Relationship

```text
                OFFLINE TRAINING FLOW

Raw Dataset
    |
    v
+----------------------+
| Data Loading Layer   |
| - read datasets      |
| - validate schema    |
+----------------------+
    |
    v
+----------------------+
| Preprocessing Layer  |
| - clean text         |
| - standardize input  |
+----------------------+
    |
    v
+----------------------+
| Training Layer       |
| - fit model          |
| - evaluate           |
| - prepare artifacts  |
+----------------------+
    |
    v
+----------------------+
| Artifacts Layer      |
| - model files        |
| - vectorizers        |
| - encoders/metadata  |
+----------------------+
    |
    v
Used by runtime inference
```

```text
                ONLINE INFERENCE FLOW

Incoming Request
    |
    v
API Route
    |
    v
Shared Preprocessing
    |
    v
Load Saved Artifacts
    |
    v
Run Prediction
    |
    v
Return Structured Result
```

### Layer Responsibilities

#### 1. API Layer

The API layer is the public entry point of the system. It is responsible for exposing endpoints, validating request payloads, invoking backend logic, and returning structured JSON responses. In the current project, this is represented by the FastAPI application and route modules such as health, version, and prediction.

#### 2. Preprocessing Layer

The preprocessing layer is the shared transformation layer between raw input and model-ready input. It is responsible for cleaning text, normalizing values, and applying reusable preparation logic. This layer is intentionally model-agnostic so the same preprocessing components can support multiple future models.

#### 3. Data Loading and Preparation Layer

This layer handles dataset ingestion for training workflows. It loads raw data, checks whether required columns exist, applies preprocessing where needed, and produces clean training inputs. It forms the bridge between raw project data and the model training pipeline.

#### 4. Training Layer

The training layer is the offline model-building component. It trains the first classical ML model, evaluates performance, and produces artifacts that can later be reused during runtime inference. This keeps model creation separate from model serving.

#### 5. Inference Layer

The inference layer is the runtime prediction engine. It loads previously saved artifacts, applies the shared preprocessing pipeline to incoming requests, runs the model, and converts predictions into API-ready outputs.

#### 6. Artifacts Layer

The artifacts layer stores all reusable training outputs required for repeatable inference. This can include trained models, vectorizers, encoders, and metadata. By persisting these artifacts, the serving layer can operate consistently without retraining a model on every startup.

### Architecture Map

```text
thendral_core_api/
|
+-- app/
|   |
|   +-- main.py                  -> FastAPI application entry point
|   +-- api/
|   |   |
|   |   +-- routes/
|   |       +-- health.py        -> health checks
|   |       +-- version.py       -> API/version metadata
|   |       +-- predict.py       -> prediction endpoint
|   |
|   +-- ml/
|       |
|       +-- config.py            -> ML/runtime configuration
|       +-- preprocess.py        -> shared preprocessing logic
|       +-- data_loader.py       -> dataset loading and validation
|       +-- train.py             -> offline training workflow
|       +-- inference.py         -> runtime prediction logic
|       +-- artifacts/           -> saved models and supporting files
|
+-- data/
|   +-- raw/                     -> source datasets
|   +-- processed/               -> cleaned/derived datasets
|
+-- requirements.txt             -> Python dependencies
+-- Dockerfile                   -> containerization setup
+-- README.md                    -> project and architecture guide
```

### Request Lifecycle

```text
1. A client sends input text to the prediction API.
2. The API layer validates the request schema.
3. The request is passed into shared preprocessing logic.
4. The inference layer loads the trained artifacts.
5. The model generates a prediction from the processed input.
6. The API returns a structured response to the client.
```

### Why This Architecture Matters

- It separates API, training, preprocessing, and inference concerns.
- It keeps shared logic reusable across future ML and AI components.
- It supports incremental growth from classical ML to larger AI workflows.
- It avoids building throwaway prototype code that must later be replaced.
- It creates a base that can evolve into a broader intelligence platform.

---

## 🧠 What is Thendral?

**Thendral** is being designed as an end-to-end intelligent AI system — not just a single model, not just an API, and not just an assistant.

The vision for Thendral is to evolve into a system that can:

- understand text, documents, websites, emails, and structured data
- process context intelligently through reusable pipelines
- use multiple models for different kinds of tasks
- grow from classical ML to deep learning, RAG, and agent-based workflows
- act as a modular intelligence platform rather than a one-off project

The long-term goal is to build Thendral as a **scalable AI architecture** where each phase becomes a permanent building block for future capabilities.

---

## 🚀 What Phase 1 Means

Phase 1 is not the final intelligence of Thendral.

Phase 1 is the **foundation layer**.

This phase is focused on building:

- a clean backend service
- reusable preprocessing pipelines
- the first machine learning workflow
- a modular architecture that can support future models
- a structure that is production-minded from the start

The purpose of Phase 1 is to make sure Thendral begins with the right engineering principles:
- separation of concerns
- reusability
- scalability
- extensibility

This phase is the first step toward making Thendral a system that can later support:
- Logistic Regression
- Linear Regression
- Neural Networks
- RAG pipelines
- Agentic AI workflows
- autonomous automation systems

---

## 🎯 My Vision for Thendral

I do not want Thendral to become a collection of disconnected scripts or isolated models.

I want Thendral to become a **highly intelligent, modular AI ecosystem** where every part has a clear role and every phase strengthens the next.

I expect Thendral to eventually:

- support multiple intelligence layers
- use shared preprocessing and data pipelines across models
- expose stable APIs for intelligence services
- integrate classical ML, deep learning, and LLM-based systems
- evolve into a powerful personal and production-grade AI platform

Every model or API built in this project should be a **stepping stone**, not throwaway code.

That means:
- shared code should stay reusable
- data pipelines should support future models
- architecture should not need to be rewritten every time the system grows

---

## 🧩 What Phase 1 Is Building

Phase 1 establishes the first working backend and ML-ready architecture for Thendral.

It focuses on:

- FastAPI as the serving layer
- reusable text preprocessing
- reusable data loading and preparation
- first classical ML model integration
- clean inference flow
- structured API endpoints

The immediate objective is to create a backend service that can:
- receive input text
- preprocess it using shared logic
- send it to a model
- return structured output through an API

This becomes the first real intelligence service inside Thendral.

---

## 🏗️ Architecture Philosophy

The architecture of Phase 1 is designed with long-term AI system growth in mind.

### Core principles

- **Separation of concerns**  
  API logic, preprocessing, training, and inference are kept separate.

- **Reusability**  
  Shared components such as preprocessing and data loading are designed to be reused across future models.

- **Extensibility**  
  New models should be added without breaking the system structure.

- **Production-minded design**  
  The system is organized as if it will continue growing into a real platform.

- **Stepping-stone development**  
  Every file and every module should remain useful in future phases.

---

## 🧱 Phase 1 Architecture

The architecture is divided into layers:

### 1. API Layer
This is the entry point into the system.

Responsibilities:
- receive requests
- validate inputs
- call backend logic
- return structured responses

### 2. Preprocessing Layer
This is the shared intelligence preparation layer.

Responsibilities:
- clean raw input
- normalize text
- prepare data for models
- remain reusable for future ML and neural models

### 3. Data Loading / Preparation Layer
This is the bridge between raw data and model-ready data.

Responsibilities:
- load datasets
- validate required columns
- apply preprocessing
- return clean training inputs

### 4. Training Layer
This is the offline model-building layer.

Responsibilities:
- train models
- evaluate models
- save trained artifacts

### 5. Inference Layer
This is the runtime model-serving layer.

Responsibilities:
- load saved model artifacts
- preprocess incoming requests
- run predictions
- return structured outputs

### 6. Artifacts Layer
This stores reusable model outputs.

Responsibilities:
- save trained vectorizers
- save trained models
- save encoders / metadata
- make inference repeatable

---

## 📁 Project Structure

```text
thendral_core_api/
├── app/
│   ├── main.py
│   ├── api/
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py
│   │       ├── version.py
│   │       └── predict.py
│   └── ml/
│       ├── __init__.py
│       ├── config.py
│       ├── preprocess.py
│       ├── data_loader.py
│       ├── train.py
│       ├── inference.py
│       └── artifacts/
├── data/
│   ├── raw/
│   └── processed/
├── requirements.txt
├── Dockerfile
└── README.md
