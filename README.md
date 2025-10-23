# Social Hub for UIC

A recommendation system to help commuter students at UIC build social connections and find study partners.

## Project Overview

This project addresses the social isolation challenges faced by commuter students at UIC by developing a machine learning-powered platform with two main features:

1. **Friend Finder**: Recommends peers with similar academic backgrounds, interests, or course enrollments
2. **Study Buddy Matcher**: Suggests classmates for study groups based on course overlap and academic goals

## Technical Approach

- **Graph Neural Networks** (GCN/GraphSAGE) for modeling student connections
- **Collaborative Filtering** for recommendation refinement
- **Embedding-based representation learning** for semantic similarity
- **AI Agent** with natural language interface

## Project Structure

```
SocialHub/
├── src/
│   ├── data/           # Data processing modules
│   ├── models/         # ML model implementations
│   ├── utils/          # Utility functions
│   └── agents/         # AI agent components
├── notebooks/          # Jupyter notebooks for exploration
├── data/
│   ├── raw/           # Original datasets
│   └── processed/     # Cleaned and processed data
├── results/           # Model outputs and evaluations
├── docs/              # Documentation
└── requirements.txt   # Python dependencies
```

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Datasets**:
   - Students' Social Network Profile Clustering Dataset (Kaggle)
   - Synthetic Student Profiles Dataset (Kaggle)

3. **Run Initial Data Exploration**:
   ```bash
   jupyter notebook notebooks/01_data_exploration.py
   jupyter notebook notebooks/02_detailed_visualizations.py
   ```

## Milestones

- [x] Project Planning and Proposal
- [ ] Dataset Collection and Preprocessing
- [ ] Data Preparation and Feature Engineering
- [ ] Baseline Model Implementation
- [ ] Advanced Model Development
- [ ] AI Agent Integration
- [ ] System Evaluation and Testing
- [ ] Final Project Presentation

