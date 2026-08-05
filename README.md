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

## 📂 Repository Structure

```text
Caption-Generation-Pipeline/
│
├── configs/                         # YAML configuration files
│   ├── dataset.yaml
│   ├── model.yaml
│   ├── ontology.yaml
│   ├── rules.yaml
│   ├── training.yaml
│   └── inference.yaml
│
├── demo/                            # Training, inference and demonstration scripts
│
├── docs/
│   ├── architecture.md
│   └── stage_reports/
│       ├── stage1.md
│       ├── stage2.md
│       ├── ...
│       └── stage9.md
│
├── outputs/
│   ├── figures/
│   ├── encoder_weights.pt
│   ├── uncertainty_head_weights.pt
│   └── evaluation_reports/
│
├── src/
│   └── hsi_caption/
│       ├── stage1_preprocessing/
│       ├── stage2_multi_encoder/
│       ├── stage3_knowledge_engine/
│       ├── stage4_reasoning/
│       ├── stage5_ontology/
│       ├── stage6_facts/
│       ├── stage7_uncertainty/
│       ├── stage8_verification/
│       ├── stage9_caption/
│       ├── config.py
│       ├── datatypes.py
│       ├── pipeline.py
│       ├── nn_utils.py
│       └── torch_modules.py
│
├── tests/
│   ├── conftest.py
│   ├── test_integration.py
│   ├── test_stage1_preprocessing.py
│   ├── test_stage2_multi_encoder.py
│   ├── test_stage3_knowledge_engine.py
│   ├── test_stage4_reasoning.py
│   ├── test_stage5_ontology.py
│   ├── test_stage6_facts.py
│   ├── test_stage7_uncertainty.py
│   ├── test_stage8_verification.py
│   └── test_stage9_caption.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 🛠️ Technology Stack

| Category | Technologies |
|-----------|--------------|
| Programming Language | Python 3.10+ |
| Deep Learning | PyTorch |
| Scientific Computing | NumPy |
| Graph Processing | NetworkX |
| Configuration | YAML |
| Data Processing | SciPy, Pandas |
| Visualization | Matplotlib |
| Testing | PyTest |
| Model Training | PyTorch Training Pipeline |
| Uncertainty Estimation | Monte Carlo Dropout |
| Knowledge Representation | Knowledge Graphs |
| Semantic Reasoning | Ontology-Based Reasoning |
| Explainable AI | Rule-Based Symbolic Reasoning |
| Version Control | Git & GitHub |

---

# 📦 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/Caption-Generation-Pipeline.git

cd Caption-Generation-Pipeline
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Verify Installation

```bash
python --version

pip list
```

---

# 📊 Dataset

This project is designed for the **WHU-Hi LongKou Hyperspectral Dataset**.

The preprocessing pipeline automatically performs:

- Dataset loading
- Integrity validation
- Spectral normalization
- Patch extraction
- Dataset inspection
- False-color visualization
- Ground truth visualization
- Official train/test split loading
- Synthetic dataset fallback (for development)

---

# ⚙️ Configuration

All experiment parameters are controlled through YAML configuration files.

Configuration includes:

- Dataset paths
- Model parameters
- Patch size
- Batch size
- Learning rate
- Number of epochs
- Knowledge graph settings
- Ontology settings
- Bayesian inference parameters
- Verification thresholds
- Caption generation options

This design enables reproducible experiments without modifying source code.

---

# 🚀 Running the Pipeline

## Stage-wise Execution

Each stage can be executed independently for experimentation and debugging.

```bash
python demo/run_stage1.py

python demo/run_stage2.py

python demo/run_stage3.py

...

python demo/run_stage9.py
```

---

## End-to-End Pipeline

```bash
python demo/run_pipeline.py
```

The complete workflow performs:

1. Data preprocessing
2. Multi-encoder feature extraction
3. Knowledge-guided reasoning
4. Expert symbolic reasoning
5. Ontology enrichment
6. Structured fact generation
7. Bayesian uncertainty estimation
8. Fact verification
9. Confidence-aware caption generation

---

# 📈 Results

The proposed Caption Generation Pipeline demonstrates the effectiveness of integrating deep learning with knowledge-guided reasoning, ontology enrichment, Bayesian uncertainty estimation, and multi-stage fact verification.

## Performance Improvements

| Metric | Before Training | After Training |
|---------|----------------:|---------------:|
| Material Match Accuracy | **79.0%** | **86.0%** |
| Mean Confidence Score | **0.002** | **0.657** |
| Mean Verification Score | **0.508** | **0.577** |

The trained encoder and Bayesian uncertainty head significantly improve prediction confidence while maintaining interpretable reasoning throughout the pipeline.

---

## Model Evaluation

The encoder network is evaluated using multiple classification metrics.

| Metric | Value |
|----------|-------|
| Overall Accuracy (OA) | **91.3%** |
| Cohen's Kappa | **0.889** |
| Evaluation Method | Official Train/Test Split |

---

# 📊 Output Visualizations

The pipeline automatically generates visualizations to support model evaluation and explainability.

## Dataset Visualizations

- False Color Composite Image
- Ground Truth Label Map
- Mean Spectral Signatures

## Model Evaluation

- Confusion Matrix
- Training Performance
- Before vs After Comparison

## Confidence Analysis

- Confidence Score Distribution
- Verification Confidence Analysis
- Bayesian Uncertainty Estimation

## Caption Generation

- Confidence-aware captions
- Explainable reasoning summary
- Structured fact visualization

> Place generated figures inside the `outputs/figures/` directory.

---

# 🧠 Explainable AI Pipeline

Unlike traditional hyperspectral classification systems, every generated caption is supported by explicit evidence.

The reasoning process consists of:

```text
Raw HSI Patch
      │
      ▼
Deep Feature Extraction
      │
      ▼
Knowledge Retrieval
      │
      ▼
Rule-based Reasoning
      │
      ▼
Ontology Matching
      │
      ▼
Structured Fact Generation
      │
      ▼
Bayesian Confidence Estimation
      │
      ▼
Fact Verification
      │
      ▼
Confidence-aware Caption
```

Every generated caption is backed by:

- Spectral evidence
- Knowledge graph evidence
- Expert rules
- Ontology reasoning
- Bayesian confidence
- Verification score

This makes the entire prediction process interpretable and explainable.

---

# 🔬 Research Contributions

The proposed framework introduces several research-oriented contributions:

- End-to-end hyperspectral image caption generation pipeline.
- Integration of spectral–spatial deep learning with symbolic reasoning.
- Knowledge-guided semantic representation learning.
- Ontology-enriched concept refinement.
- Bayesian uncertainty estimation using Monte Carlo Dropout.
- Confidence-aware natural language generation.
- Multi-source fact verification before caption generation.
- Modular architecture supporting independent stage development and testing.
- Explainable AI framework for trustworthy hyperspectral image understanding.

---

# 🧪 Testing

The project includes comprehensive testing to ensure correctness and reproducibility.

## Unit Tests

- Stage 1 – Data Preprocessing
- Stage 2 – Multi-Encoder
- Stage 3 – Knowledge Engine
- Stage 4 – Expert Reasoning
- Stage 5 – Ontology Engine
- Stage 6 – Structured Fact Generation
- Stage 7 – Bayesian Uncertainty
- Stage 8 – Fact Verification
- Stage 9 – Caption Generation

## Integration Tests

- End-to-end pipeline execution
- Official dataset split validation

Run all tests using:

```bash
pytest tests/
```

Or execute an individual stage test:

```bash
pytest tests/test_stage3_knowledge_engine.py
```

---

# 🔮 Future Work

Potential extensions include:

- Large Vision-Language Model (VLM) integration
- Graph Neural Networks for knowledge reasoning
- Multi-temporal hyperspectral caption generation
- Multi-sensor data fusion (HSI + LiDAR + SAR)
- Active learning using uncertainty estimation
- Retrieval-Augmented Caption Generation (RAG)
- Multilingual caption generation
- Deployment as a cloud-based inference service
- Real-time UAV hyperspectral caption generation

---

# 🤝 Contributing

Contributions are welcome.

If you would like to improve the project:

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

# 📜 License

This project is released under the **MIT License**.

See the `LICENSE` file for additional details.

---

# 🙏 Acknowledgements

This project builds upon concepts from:

- Hyperspectral Image Analysis
- Deep Learning
- Explainable Artificial Intelligence (XAI)
- Neuro-Symbolic AI
- Knowledge Graphs
- Ontology Engineering
- Bayesian Deep Learning
- Trustworthy AI

Special thanks to the researchers and open-source community whose work has contributed to the advancement of hyperspectral image understanding.

---

# 📧 Contact

For questions, discussions, or collaboration opportunities, feel free to open an issue or contact the repository maintainer.

---

<div align="center">

**⭐ If you find this project useful, consider giving the repository a star! ⭐**

</div>
