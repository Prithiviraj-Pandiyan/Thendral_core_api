# Thendral Core API

Phase 1 of Thendral — foundational backend service for your personal AI system.

## Overview

Thendral Core API is the first backend module in Project Thendral.  
This phase focuses on building a clean FastAPI service structure that will later support ML inference, email intelligence, and agent workflows.

## Current Features

- Health endpoint
- Version endpoint
- Prediction placeholder endpoint
- Modular routing structure
- GitHub-ready project setup

## Project Structure

```text
app/
├── main.py
└── api/
    └── routes/
        ├── __init__.py
        ├── health.py
        ├── predict.py
        └── version.py