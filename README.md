# GENAI Assignment - Implementation

## Overview

This repo implements a comprehensive Machine Learning and NLP assignment with **standout features** that go beyond basic requirements:


##  Features Implemented

### Part 1: Gaussian Mixture Model
- ✅ Probabilistic generative model implementation
- ✅ Data generation and visualization
- ✅ Performance metrics (BIC, AIC, Silhouette Score)
- ✅ Gaussian contour visualization
- ✅ New sample generation from learned distribution

### Part 2: Markov Chain Text Generator
- ✅ Configurable order (1st, 2nd, 3rd order)
- ✅ Statistical analysis of chain
- ✅ Text generation with seed support
- ✅ Transition probability tracking

### Part 3: Prompt Engineering
- ✅ **Interview Approach** - Socratic questioning method
- ✅ **Chain of Thought (CoT)** - Step-by-step reasoning
- ✅ **Tree of Thought (ToT)** - Multi-path exploration
- ✅ Comprehensive comparison and analysis

### Part 4: Shot-based Prompting
- ✅ **Zero-shot** - Direct problem solving
- ✅ **Few-shot** - Learning from examples
- ✅ Performance comparison visualization

### Part 5: Advanced Features (🌟 BONUS)
- ✅ Gemini API integration for real AI responses
- ✅ Professional visualizations (publication quality)
- ✅ Multiple export formats (JSON, CSV, TXT)
- ✅ Comprehensive statistical analysis
- ✅ Detailed performance metrics
- ✅ Automated report generation

---

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Get Gemini API Key

```bash
export GEMINI_API_KEY='your-api-key-here'
```

Or modify the code:
```python
# In ml_nlp_assignment.py, line ~600
api_key = 'your-api-key-here'  # Replace None with your key
```

---

## Usage

### Run the Complete Assignment

```bash
python ml_nlp_assignment.py
```

### What Happens:

1. **GMM Training & Visualization**
   - Generates synthetic data
   - Trains Gaussian Mixture Model
   - Creates visualization with clusters and contours
   - Outputs performance metrics

2. **Markov Chain Text Generation**
   - Tests 1st, 2nd, and 3rd order chains
   - Generates sample text
   - Analyzes chain statistics

3. **Prompt Engineering Comparison**
   - Tests Interview, CoT, and ToT approaches
   - Compares prompt structures
   - Analyzes response patterns

4. **Zero-shot vs Few-shot**
   - Demonstrates both approaches
   - Visualizes differences
   - Exports comparison data

5. **Results Export**
   - Saves visualizations as PNG
   - Exports metrics as JSON/CSV
   - Generates comprehensive report

---

##  Output Files

After running, you'll find in the `output/` directory:

```
output/
├── gmm_visualization.png          # GMM clusters and contours
├── prompt_comparison.png          # Prompt engineering comparison
├── gmm_metrics.json              # GMM performance metrics
├── gmm_metrics.csv               # GMM metrics (spreadsheet format)
├── markov_stats.json             # Markov chain statistics
└── comprehensive_report.txt      # Full detailed report
```

---




##  Comparison Summary

### Prompt Engineering Approaches

| Approach | Complexity | Best For | Example Use Case |
|----------|-----------|----------|------------------|
| Interview | Medium | Education, Exploration | Teaching, Tutoring |
| CoT | High | Logic, Math | Problem-solving, Proofs |
| ToT | Very High | Strategy, Planning | Business decisions |
| Zero-shot | Low | Quick answers | General queries |
| Few-shot | Medium | Pattern learning | Specific formats |

---

## Customization

### Change GMM Parameters:
```python
gmm = EnhancedGMM(
    n_components=5,      # Number of clusters
    n_samples=1000       # Number of data points
)
```

### Change Markov Order:
```python
markov = MarkovChainGenerator(order=3)  # Try 1, 2, or 3
```

### Use Your Own Text Corpus:
```python
with open('your_text.txt', 'r') as f:
    text = f.read()
markov.train(text)
```

### Add Your Own Problems:
```python
problem = "Your custom problem here"
results = comparator.compare_all_approaches(problem)
```



---

## 📝 License

This project is for educational purposes.


