# Caption Generation Pipeline

> **A Knowledge-Guided Neuro-Symbolic Framework for Explainable Hyperspectral Image Caption Generation with Bayesian Uncertainty Estimation and Multi-Stage Fact Verification**

## 📖 Overview

**Caption Generation Pipeline** is an end-to-end framework for generating **trustworthy, explainable natural-language captions** from hyperspectral imagery (HSI). Unlike conventional HSI classification systems that only predict land-cover labels, this project combines **deep learning**, **domain knowledge**, **ontology reasoning**, **Bayesian uncertainty estimation**, and **symbolic fact verification** to generate semantically rich, confidence-aware image descriptions.

The framework follows a **nine-stage modular pipeline**, progressively transforming raw hyperspectral image patches into verified natural-language captions through spectral–spatial feature extraction, knowledge-guided reasoning, ontology enrichment, structured fact generation, uncertainty estimation, and explainable caption synthesis.

Designed with a configuration-driven architecture, the project supports reproducible experimentation, modular development, and easy replacement of individual pipeline components while maintaining a consistent end-to-end workflow.

---

## 🎯 Objectives

The primary objectives of this project are:

- Generate natural-language captions from hyperspectral imagery.
- Integrate deep learning with symbolic reasoning.
- Improve interpretability using expert knowledge and ontologies.
- Estimate prediction uncertainty using Bayesian inference.
- Verify generated facts before caption generation.
- Produce confidence-aware and explainable captions.
- Provide a modular research framework for hyperspectral image understanding.

---

## ✨ Key Features

- Multi-stage modular architecture
- Spectral–Spatial Multi-Encoder Network
- Knowledge-guided spectral retrieval
- Knowledge Graph construction and reasoning
- Expert rule-based symbolic reasoning
- Ontology-enriched semantic representation
- Structured RDF-style fact generation
- Bayesian uncertainty estimation using Monte Carlo Dropout
- Confidence calibration
- Multi-source fact verification
- Confidence-aware natural-language generation
- Explainable AI (XAI)
- Configuration-driven YAML architecture
- Lightweight NumPy reference implementation
- PyTorch training pipeline
- Comprehensive unit and integration testing

---

## 🚀 Highlights

- End-to-end **9-stage hyperspectral image caption generation pipeline**
- Neuro-symbolic AI architecture
- Bayesian uncertainty estimation
- Explainable decision making
- Knowledge-guided semantic reasoning
- Ontology-based concept refinement
- Structured fact verification
- Research-oriented modular software design
- Reproducible experimental framework

## 🏗️ System Architecture

The Caption Generation Pipeline follows a modular **nine-stage architecture**, where each stage performs a dedicated task while passing structured outputs to the next stage. This design promotes modularity, reproducibility, maintainability, and independent testing.

```text
                           ┌───────────────────────────┐
                           │  Stage 1                  │
                           │ Data Preprocessing        │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │  Stage 2                  │
                           │ Multi-Encoder Network     │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │  Stage 3                  │
                           │ Spectral Knowledge Engine │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │  Stage 4                  │
                           │ Expert Reasoning Layer    │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │  Stage 5                  │
                           │ Ontology Engine           │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │  Stage 6                  │
                           │ Structured Fact Generator │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │  Stage 7                  │
                           │ Bayesian Uncertainty      │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │  Stage 8                  │
                           │ Fact Verification         │
                           └─────────────┬─────────────┘
                                         │
                                         ▼
                           ┌───────────────────────────┐
                           │  Stage 9                  │
                           │ Caption Generation        │
                           └───────────────────────────┘
```

---

## ⚙️ Pipeline Stages

### Stage 1 — Data Preprocessing

Prepares the hyperspectral dataset for downstream processing.

**Responsibilities**

- Dataset loading
- Dataset validation
- Spectral normalization
- Dataset inspection
- Patch extraction
- Official train/test split loading
- False-color visualization
- Ground-truth visualization

**Output**

- Preprocessed hyperspectral patches

---

### Stage 2 — Multi-Encoder Network

Extracts complementary information from hyperspectral patches using multiple encoders.

**Components**

- Spectral CNN Encoder
- Spatial Vision Transformer Encoder
- Metadata Encoder
- Attention-based Feature Fusion

**Output**

- Unified spectral-spatial feature embedding

---

### Stage 3 — Spectral Knowledge Engine

Transforms learned features into knowledge-guided semantic representations.

**Components**

- Spectral Library Retrieval
- Material Identification
- Knowledge Graph Construction
- Semantic Embedding Generation

**Output**

- Knowledge-aware semantic representation

---

### Stage 4 — Expert Reasoning Layer

Applies symbolic reasoning using expert-defined rules.

**Components**

- Rule Engine
- Constraint Checker
- Context Generator
- Reasoning Confidence Estimation

**Output**

- Context-aware symbolic knowledge

---

### Stage 5 — Ontology-Enriched Semantic Generation

Maps symbolic knowledge into a formal ontology.

**Components**

- Ontology Loader
- Ontology Matcher
- Semantic Generator
- Ontology Refiner

**Output**

- Ontology-enriched semantic representation

---

### Stage 6 — Structured Fact Generation

Converts symbolic knowledge into machine-readable facts.

**Components**

- Fact Extraction
- Triplet Generation
- Relationship Generation
- Fact Ranking

**Output**

- Ranked RDF-style structured facts

---

### Stage 7 — Bayesian Uncertainty Estimation

Estimates predictive uncertainty using Bayesian Deep Learning.

**Components**

- Bayesian Predictor
- Monte Carlo Dropout
- Entropy Decomposition
- Temperature Calibration
- Confidence Fusion

**Output**

- Confidence score
- Predictive uncertainty
- Epistemic uncertainty
- Aleatoric uncertainty

---

### Stage 8 — Fact Verification

Verifies generated facts before natural-language generation.

**Verification Sources**

- Semantic similarity
- Knowledge graph consistency
- Ontology consistency
- Rule consistency
- Bayesian classifier agreement

**Output**

- Verified facts
- Rejected facts
- Verification confidence

---

### Stage 9 — Caption Generation

Generates explainable natural-language captions from verified facts.

**Components**

- Template Selection
- Caption Generation
- Caption Refinement
- Explanation Generation
- Language Controller

**Output**

- Confidence-aware natural-language caption
- Explainable reasoning summary
