# Probabilistic & AI-Driven Analysis of Public Transport Service Interruptions
BTN (Bayesian Trust Networks) + Generative AI toolkit for modeling, simulating, and forecasting congestion, breakdowns, and delays in urban public transport. The project combines Bayesian networks (for causal/uncertainty reasoning) with GAN-based synthetic scenario generation to improve prediction quality and stress-test operational strategies.

# Key Features
Bayesian Trust Network (BTN) to capture dependencies between factors (e.g., reason → delay duration, borough, school age group).  
Generative scenarios (GANs) to augment rare events and create realistic peak/emergency cases for robust training.  
Reproducible pipeline from data prep → model training → JSON export of the learned BTN.  
Example artifacts included:  
bayesian_network.json — learned network structure + CPDs (conditional probability tables).  
data.csv — sample for quick start (try taking a smaller portion to start with).  

# Getting Started
1) Environment  
Python 3.9+ is recommended  
python -m venv .venv  
source .venv/bin/activate     # Windows: .venv\Scripts\activate  
Install deps (minimal set)  
pip install numpy pandas scikit-learn tensorflow keras matplotlib  
Notes  
• The GAN/BTN examples in the paper use Keras/TensorFlow layers for generator/discriminator prototypes.  
• If you prefer PyTorch, porting the simple MLP generator/discriminator is straightforward.  

2) Data  
This project references the NYC Bus Breakdown and Delays dataset (public/open). You can download full data from Kaggle and replace/extend data.csv:  
Kaggle: NY Bus Breakdown and Delays (NYC DOE) — 2015+ feed.  
Place your CSV next to model.py (or update the path in code if you keep a separate data/ folder).  

3) Run
python model.py  
What it does (default flow):  
Load & preprocess data (e.g., parse categorical columns, normalize durations).  
Train simple GAN to generate synthetic samples for under-represented cases (accidents, severe delays, holidays, etc.).  
Fit models (e.g., Random Forest / Logistic Regression) on historical vs. historical+synthetic to compare impact.  
Export BTN as bayesian_network.json (nodes, edges, CPDs) for downstream decision support.  
The script prints basic metrics and writes the JSON BTN export for integration.  
Typical Use Cases  
What-if scenario testing (peak loads, weather shocks, events).  
Causal reasoning under uncertainty (BTN-based inference).  
Augment scarce labels (synthetic rare events) to improve minority-class performance.  
Integration into dispatching/decision-support tools (consume bayesian_network.json).  
