# KDraft Backend

> **Portfolio Project** - AI-Powered Draft Generation System

A machine learning backend exploring language model training and deployment with FastAPI.

---

## ✨ Overview

KDdraft is an AI-powered backend that generates competitive League of Legends drafts using fine-tuned transformer models trained on real match data.

The system predicts coherent and meta-relevant picks and bans given a partial draft state.

Input: "[AS_TEAM],Gen.G,[VS_TEAM],Hanwha Life Esports,[SIDE],RED,[PATCH],25.17,<BOS>,[BLUE_BAN1],Galio,[RED_BAN1],"

Output: Full completed draft

## 📊 Results

- Eval loss: 0.82
- Dataset: 100k+ professional games
- Model: GPT-2-small fine-tuned

The model is able to generate coherent drafts respecting role distribution and draft constraints.

## 🎓 Learning Project

This project was developed as part of my AI specialization studies (Master's in Computer Science). As a learning experience, the focus was on:

- **Implementing ML pipelines** from scratch (tokenization, training, fine-tuning)
- **Building a REST API** with FastAPI
- **Integrating transformers** for NLP tasks
- **Hands-on experience** with PyTorch and HuggingFace

### Areas for Improvement

As this is a learning project, there are several refactoring opportunities:
- Refactor into a modular, package-based architecture
- Add unit and integration tests (pytest)
- Improve configuration handling (env-based / Hydra)
- Implement model versioning and experiment tracking (MLflow)

**I'm aware of these limitations and actively learning best practices for production-grade ML systems.**

---

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **Language**: Python
- **ML/AI**: PyTorch, Transformers (HuggingFace)
- **Training**: Custom training and fine-tuning pipelines

---

## 🚀 What I Built

### Core Features

- **Custom Tokenizer Creation**: Built tokenization pipeline for domain-specific text
- **Model Training Pipeline**: Implemented training loop from scratch
- **Fine-tuning Workflow**: Adapted pre-trained models to specific tasks
- **REST API**: FastAPI endpoints for model inference
- **Draft Generation**: End-to-end text generation system

### Technical Skills Demonstrated

✅ Deep Learning with PyTorch  
✅ NLP with HuggingFace Transformers  
✅ API Development with FastAPI  
✅ ML Pipeline Design  
✅ Model Training & Fine-tuning

---

## 📁 Project Structure

```
kdraft-backend/
├── backend/              # FastAPI application
│   └── main.py          # API entry point
├── src/                 # Source code modules
├── resources/           # Training data and resources
├── create_tokenizer.py  # Tokenizer creation script
├── train_model.py       # Model training pipeline
├── finetune_model.py    # Fine-tuning script
├── test_model.py        # Model evaluation
└── generate_draft.py    # Draft generation utility
```

---

## 🧠 ML Pipeline

The project implements a complete machine learning workflow:

### 1. Tokenizer Creation
```
python create_tokenizer.py
```
Creates a custom tokenizer adapted to the training data.

### 2. Model Training
```
python train_model.py
```
Trains a language model from scratch or from a checkpoint.

### 3. Fine-tuning
```
python finetune_model.py
```
Fine-tunes pre-trained models for specific draft generation tasks.

### 4. Evaluation
```
python test_model.py
```
Evaluates model performance on test data.

### 5. Inference
```
python generate_draft.py
```
Generates drafts using the trained model.

---

## 🔌 API Usage

### Starting the Server

```
fastapi run backend/main.py
```

The API runs locally at `http://localhost:8000`


### POST /generate

Request:

```json
{
  "AS_TEAM": "Gen.G",
  "VS_TEAM": "HLE",
  "SIDE": "RED",
  "PATCH": "25.17",
  "draft_sequence": "[BLUE_BAN1],Galio"
}
```

Response:
```json
[
  {"token": "X", "probability": 0.42},
  {"token": "Y", "probability": 0.21}
]
```

## 🎯 Learning Outcomes

Through this project, I gained practical experience with:

- **Transformer architectures** and attention mechanisms
- **Training dynamics** (learning rates, batch sizes, gradient accumulation)
- **Fine-tuning strategies** for transfer learning
- **API design** for ML model serving
- **Resource management** for GPU training
- **Debugging ML pipelines** and handling edge cases

## 👤 Author

**Pierre Blondeau**

🎓 Master's Degree (Bac+5) in Computer Science  
🤖 Specialization in Artificial Intelligence  
💼 Actively seeking opportunities in ML/AI Engineering

## 💡 Feedback Welcome

This is a learning project, and I welcome constructive feedback! If you're a recruiter or developer reviewing this code, I'd appreciate any suggestions on:

- Architecture improvements
- Code quality enhancements
- Best practices I should adopt
- Resources for further learning

---

**📌 This repository is part of my portfolio and demonstrates my learning journey in AI/ML engineering.*