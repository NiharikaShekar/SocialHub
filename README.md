# Social Hub for UIC

A comprehensive ML-powered recommendation system to help commuter students at UIC build social connections, find study partners, and discover campus events.

## Project Overview

This project addresses the social isolation challenges faced by commuter students at UIC by developing a machine learning-powered platform with **three main features**:

1. **Friend Finder**: Recommends peers with similar academic backgrounds, interests, or course enrollments
2. **Study Buddy Matcher**: Suggests classmates for study groups based on course overlap and academic goals
3. **Event Recommendation System**: Recommends UIC campus events based on student interests and preferences

## Features

### 1. Friend Finder
- Matches students based on hobbies, interests, and demographics
- Uses clustering algorithms (K-means, Spectral Clustering) and embedding-based similarity
- Advanced Graph Neural Networks (GraphSAGE) for link prediction
- Collaborative Filtering for recommendation refinement

### 2. Study Buddy Matcher
- Matches students based on course overlap and schedule compatibility
- Considers GPA, major, and academic year
- Content-based filtering with semantic embeddings
- Optimized clustering for similar study patterns

### 3. Event Recommendation System
- Scrapes real-time events from UIC Today events page
- Content-based filtering using Sentence-BERT embeddings
- Semantic similarity matching between student profiles and events
- Recommends events by category, location, and student interests

## Technical Stack

### Machine Learning Models

#### Baseline Methods
- **Clustering**: K-means, Spectral Clustering for grouping similar students
- **Embedding-based**: Sentence-BERT (all-MiniLM-L6-v2) for semantic representations
- **Cosine Similarity**: For matching student profiles

#### Advanced Models
- **Graph Neural Networks (GNN)**:
  - GraphSAGE for heterogeneous graph learning
  - Link prediction for friendship/study partnership recommendations
  - Trained on student-student and student-course graphs
  
- **Collaborative Filtering**:
  - Matrix Factorization (NMF) for user-user interactions
  - Latent factor learning for recommendation refinement
  
- **Content-Based Filtering**:
  - Semantic embeddings for event recommendations
  - Boost factors for hobby matching and academic interests

### Natural Language Processing
- **Sentence Transformers**: Semantic embeddings for text (bios, interests, event descriptions)
- **Text Mining**: Interest extraction and keyword matching

### Data Processing
- **Web Scraping**: BeautifulSoup4 for UIC events data
- **Feature Engineering**: Categorical encoding, numerical scaling, text preprocessing
- **Data Cleaning**: Age normalization, missing value handling

### APIs & Infrastructure
- **FastAPI**: REST APIs for all recommendation services
- **Google ADK**: AI Agent with natural language interface
- **Port Configuration**:
  - Study Buddy API: Port 8000
  - Friend Finder API: Port 8001
  - Event Recommendations API: Port 8002

## Evaluation Results

### Friend Finder Performance
- **Precision@5**: 98.7%
- **Recall@5**: 24.7%
- **F1-Score@5**: 39.5%
- **Precision@10**: 98.3%
- **Recall@10**: 49.1%
- **MRR**: 99.1%
- **Coverage**: 27.6%
- **Diversity**: 13.7%

### Study Buddy Performance
- **Precision@5**: 78.8%
- **Recall@5**: 19.7%
- **F1-Score@5**: 31.5%
- **Precision@10**: 67.1%
- **Recall@10**: 33.5%
- **MRR**: 94.0%
- **Coverage**: 16.1%
- **Diversity**: 31.6%

## Project Structure

```
SocialHub/
├── notebooks/                    
│   ├── 01_data_exploration.py   
│   ├── 02_detailed_visualizations.py
│   ├── 04_friend_finder_baseline_embedding.py
│   ├── 05_study_buddy_baseline_embedding.py
│   ├── 09_find_optimal_clusters.py
│   ├── 10_collaborative_filtering.py
│   ├── 11_evaluation_metrics.py
│   ├── 12_model_comparison_simple.py
│   ├── 13_validation_testing.py
│   ├── 14_events_scraper.py     
│   ├── 15_event_recommendation.py  
│   ├── 16_event_recommendations_demo.py  
│   ├── gnn_graphsage.py        
│   └── utils/
│       └── create_embeddings.py
├── friend_finder_api/            
│   ├── main.py
│   ├── engine.py
│   └── model_files/
├── study_buddy_api/              
│   ├── main.py
│   ├── engine.py
│   └── model_files/
├── event_recommendation_api/     
│   └── main.py
├── social_hub_agent/             
│   └── agent.py
├── data/
│   ├── raw/                      
│   │   ├── 03_Clustering_Marketing.csv
│   │   ├── student_profiles.jsonl
│   │   └── uic_events_raw.json
│   └── processed/                
│       ├── marketing_processed.csv
│       ├── profiles_processed.csv
│       └── uic_events_processed.csv
├── results/                      # Model outputs and evaluations
│   ├── baseline/                 # Baseline model results
│   ├── GNN/                      # Graph Neural Network results
│   ├── collaborative_filtering/  # Collaborative filtering results
│   ├── evaluation/               # Comprehensive evaluation metrics
│   ├── event_recommendations/    # Event recommendation visualizations
│   ├── model_comparison/         # Model comparison results
│   └── validation/               # Cross-validation results
├── docs/                         
└── requirements.txt              
```

## Getting Started

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd SocialHub

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

The project uses two main datasets:
- **Students' Social Network Profile Clustering Dataset** (Kaggle)
- **Synthetic Student Profiles Dataset** (Kaggle)

Place datasets in `data/raw/` directory:
- `03_Clustering_Marketing.csv`
- `student_profiles.jsonl`

Then run data preprocessing:
```bash
cd notebooks
python 01_data_exploration.py
python 03_clean_marketing_ages.py
```

### 3. Running the Recommendation Systems

#### Friend Finder
```bash
# Start the API server
cd friend_finder_api
uvicorn main:app --port 8001 --reload
```

#### Study Buddy
```bash
# Start the API server
cd study_buddy_api
uvicorn main:app --port 8000 --reload
```

#### Event Recommendations
```bash
# First, scrape events
cd notebooks
python 14_events_scraper.py

# Run the demo
python 16_event_recommendations_demo.py

# Start the API server (optional)
cd ../event_recommendation_api
uvicorn main:app --port 8002 --reload
```

#### AI Agent
```bash
# Make sure APIs are running first
cd social_hub_agent
# Use Google ADK to run the agent
python agent.py
```

## API Usage Examples

### Friend Finder API

**Endpoint**: `POST http://localhost:8001/find_friends`

```json
{
  "name": "John Doe",
  "age": 22,
  "sex": "Male",
  "major": "Computer Science",
  "year": "Junior",
  "gpa": 3.5,
  "hobbies": ["coding", "photography", "hiking"],
  "unique_quality": "Passionate about AI",
  "story": "Looking for friends with similar tech interests"
}
```

### Study Buddy API

**Endpoint**: `POST http://localhost:8000/recommend/study_buddy`

```json
{
  "name": "Jane Smith",
  "age": 20,
  "sex": "Female",
  "major": "Biology",
  "year": "Sophomore",
  "gpa": 3.8,
  "courses": ["BIOS 110", "CHEM 122", "MATH 180"],
  "free_slots": ["Mon_14:00", "Wed_14:00", "Fri_10:00"],
  "unique_quality": "Excellent at explaining concepts",
  "story": "Looking for study partner for upcoming exams"
}
```

### Event Recommendations API

**Endpoint**: `POST http://localhost:8002/recommend`

```json
{
  "student": {
    "name": "Alex Johnson",
    "major": "Computer Science",
    "hobbies": ["technology", "workshops"],
    "year": "Senior"
  },
  "top_k": 5,
  "date_range_days": 30
}
```

## Model Training & Evaluation

### Baseline Models
```bash
cd notebooks
python 04_friend_finder_baseline_embedding.py
python 05_study_buddy_baseline_embedding.py
```

### Advanced Models
```bash
# Graph Neural Networks
python gnn_graphsage.py

# Collaborative Filtering
python 10_collaborative_filtering.py
```

### Evaluation
```bash
# Comprehensive evaluation metrics
python 11_evaluation_metrics.py

# Model comparison
python 12_model_comparison_simple.py

# Cross-validation
python 13_validation_testing.py
```

### Event Recommendations Demo
```bash
python 16_event_recommendations_demo.py
```

## Results & Visualizations

All results are saved in the `results/` directory:

- **Baseline Models**: Clustering visualizations, optimal cluster analysis
- **GNN Models**: Training curves, ROC curves, Precision-Recall curves
- **Collaborative Filtering**: User clustering, latent factors, interaction matrices
- **Evaluation**: Comprehensive metrics, comparison charts
- **Event Recommendations**: 6 separate visualizations (categories, similarity scores, majors, etc.)

## Key Achievements

- **Complete ML Pipeline**: From data preprocessing to model deployment  
- **Multiple Recommendation Systems**: Friend Finder, Study Buddy, and Event Recommendations  
- **State-of-the-Art Models**: Graph Neural Networks, Collaborative Filtering, Semantic Embeddings  
- **Comprehensive Evaluation**: Multiple metrics (Precision, Recall, MRR, Coverage, Diversity)  
- **Production-Ready APIs**: FastAPI endpoints for all services  
- **AI Agent Integration**: Natural language interface for easy interaction  
- **Real-Time Data**: Web scraping for up-to-date UIC events  

## Dependencies

Key libraries used:
- **PyTorch** & **PyTorch Geometric**: Deep learning and graph neural networks
- **scikit-learn**: Machine learning algorithms
- **Sentence Transformers**: NLP embeddings
- **FastAPI**: API framework
- **Pandas & NumPy**: Data processing
- **BeautifulSoup4**: Web scraping
- **Google ADK**: AI Agent framework

See `requirements.txt` for complete list.



## License

This project is part of CS512 coursework at University of Illinois Chicago.

## Acknowledgments

- UIC Department of Computer Science
- Kaggle for providing student profile datasets
- UIC Today for events data
- Open source ML libraries and frameworks

---

