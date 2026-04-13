# ⚾ MLB Pitch Predictor - Tyler Glasnow Edition

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Cloud](https://img.shields.io/badge/cloud-Google%20Cloud-blue.svg)
![ML](https://img.shields.io/badge/ML-Vertex%20AI-orange.svg)

Machine learning model to predict Tyler Glasnow's pitch type based on game situation, trained on 4,000+ pitches using Google Cloud Vertex AI AutoML.

> **Note:** This is a portfolio/demonstration project. The model endpoint is not publicly deployed to avoid unnecessary cloud costs. Code and methodology are provided for reference and learning purposes.

## 🎯 Project Overview

This project demonstrates end-to-end machine learning pipeline from data collection through deployment and real-world testing. The model predicts pitch type (4-Seam Fastball, Slider, Curveball, Sinker) based on:
- Count (balls/strikes)
- Outs
- Inning
- Runners on base
- Batter handedness

**Built with:** Google Cloud Vertex AI | Streamlit | Python | Plotly

---

## 📊 Model Performance

### Training Metrics (2024 Data)
| Metric | Value |
|--------|-------|
| **ROC AUC** | 0.731 |
| **Log Loss** | 1.202 |
| **Best Accuracy** | 75% (Fastball predictions) |
| **Training Data** | 4,235 pitches (2024 season) |

### Real-World Testing (2026 Game)
| Metric | Value |
|--------|-------|
| **Raw Accuracy** | 51% |
| **Adjusted Accuracy** | 65% (grouping FF/SI) |
| **vs. Random Guessing** | 2x better (25% baseline) |

### Key Finding: Model Drift
Testing revealed Tyler Glasnow's pitch mix had shifted 10% between training data (2024) and production testing (2026):
- **Four-seam Fastball:** -10%
- **Curveball:** +10%
- **Sinker:** +8%
- **Slider:** -8%

**Lesson:** Production ML requires continuous monitoring and retraining as subject behavior evolves.

---

## 🔍 Feature Importance

The model learned that **strike count** is the dominant factor in pitch selection (2x more important than any other variable):

| Feature | Relative Importance |
|---------|---------------------|
| **Strikes** | 100% |
| Hitter Hand | 45% |
| Balls | 40% |
| Outs | 25% |
| Inning | 18% |
| Runner 2B | 12% |
| Runner 3B | 10% |
| Runner 1B | 8% |

This aligns with baseball strategy: pitch selection changes dramatically between 0-2 (pitcher's count) and 3-1 (hitter's count).

---

## 🛠️ Tech Stack

- **ML Platform:** Google Cloud Vertex AI (AutoML Tabular Classification)
- **Deployment:** Vertex AI Endpoints with auto-scaling (scale-to-zero enabled)
- **Web App:** Streamlit
- **Visualizations:** Plotly
- **Data Source:** Baseball Savant (Statcast)
- **Language:** Python 3.10+

---

## 🎮 Application Demo

The Streamlit interface allows interactive pitch predictions:

### Features
- **Game Situation Controls:** Set inning, count, outs, runners, and batter handedness using intuitive sliders and checkboxes
- **Real-time Predictions:** Click "Predict Pitch" to get probability distributions across all 4 pitch types
- **Visual Results:** Bar charts showing likelihood of each pitch, with top prediction highlighted
- **Game Context Display:** Summary of the current game situation for reference

### Example Scenarios Tested

**Pitcher's Count (0-2):**
- Model predicted: 72% Slider, 18% Curveball, 8% Fastball, 2% Sinker
- **Insight:** When ahead in count, Glasnow can afford to throw breaking balls

**Hitter's Count (3-1):**
- Model predicted: 68% Fastball, 22% Sinker, 8% Slider, 2% Curveball
- **Insight:** When behind in count, model correctly predicts need for strikes

**High Leverage (Bases Loaded, 2 Outs, 3-2 Count):**
- Model adjusts probabilities based on game pressure
- Demonstrates learned understanding of situational baseball

**Screenshots and demo videos available in project documentation.**

---

Training Configuration
Platform: Vertex AI AutoML Tabular
Optimization metric: Log loss
Budget: 2 node hours
Early stopping: Enabled
Training time: ~2.5 hours (wall-clock)
Technical Challenges & Solutions
Challenge 1: Data Type Mismatch

Issue: AutoML interpreted numeric columns as categorical strings
Solution: Convert all inputs to strings during inference: {k: str(v) for k, v in data.items()}
Challenge 2: Cold Start Latency

Issue: Scale-to-zero causes 30-60 second first prediction delay
Trade-off: Accept latency for cost savings vs. always-on ($0.10/hr continuous)
Challenge 3: Model Drift

Issue: 10% repertoire shift between training and testing
Solution: Planned weighted retraining with 2026 data

🧪 Real-World Testing
Test Methodology
Collected pitch-by-pitch data from Glasnow's first 2026 start (97 pitches)
Fed each game situation into the model before pitch was thrown
Compared prediction vs. actual pitch
Analyzed patterns in correct/incorrect predictions
Results Analysis
51% raw accuracy (2x better than random 25% baseline)
65% adjusted accuracy (when grouping FF/SI as "fastballs")
68% accuracy in pitcher's counts (0-2, 1-2) - model learned breaking ball strategy
61% accuracy in hitter's counts (3-1, 2-0) - correctly predicted fastballs
Model drift identified: 10% shift in pitch repertoire (2024 → 2026)
Key Learnings
Model correctly learned game context patterns (count, leverage situations)
Fastball vs. breaking ball distinction was strong (65% when grouped)
Four-seam / Sinker confusion is expected (strategically similar pitches)
Player evolution requires continuous retraining - behavior changes over time
Detailed analysis: See Medium article "I Built an AI to Predict MLB Pitches. Then Reality Hit."

🔄 Future Improvements
Planned Enhancements
 Retrain with weighted 2024+2026 data (favor recent pitches)
 Add pitch location prediction (zone-based classification)
 Expand to multiple pitchers for comparative analysis
 Implement continuous learning pipeline (retrain after every 5 games)
 Add model drift monitoring dashboard
 A/B test: AutoML vs. custom neural network
Production ML Lessons Applied
 Implement automated model performance monitoring
 Create retraining triggers (accuracy threshold, data drift detection)
 Add model versioning and A/B testing framework
 Build automated data collection pipeline
 Develop shadow deployment for new model versions

📊 Visualizations

--------
Application Interface
<img width="1699" height="893" alt="mlb-pitch-predictor:images:app_screenshot" src="https://github.com/user-attachments/assets/2c17fd40-bd8e-4ab5-bd0f-0495b23417d1" />

--------

Prediction Example
<img width="1681" height="874" alt="mlb-pitch-predictor:images:prediction_example" src="https://github.com/user-attachments/assets/8df8ee9b-4f67-4fee-84ea-add022b53529" />

Model Insights
Feature Importance
Strike count dominates decision-making (2x more important than any other factor)

🐛 Known Limitations
Cold Start Latency
Issue: First prediction after endpoint idle takes 30-60 seconds

Cause: Scale-to-zero configuration (cost optimization trade-off)

Production Solution: Set min replicas to 1 (eliminates cold start but costs $0.10/hour continuously)

Data Drift
Issue: Model trained on 2024 data doesn't perfectly predict 2026 behavior

Root Cause: Glasnow's repertoire evolved 10% (more curveballs, fewer fastballs)

Solution: Retrain with weighted recent data or implement continuous learning

Sample Size Constraints
Issue: Testing on single game (97 pitches) has high variance

Mitigation: Collecting more 2026 games for robust performance evaluation

📚 Technical Documentation

Architecture Decisions

Why AutoML vs. Custom Model?
Rapid prototyping (2.5 hours vs. days of tuning)
Automatic feature engineering
Good baseline for comparison
Cost-effective for small datasets

Why Scale-to-Zero?
Demo project with intermittent usage
90%+ cost savings vs. always-on
Acceptable latency trade-off for non-production use

Why Streamlit vs. Flask/FastAPI?
Rapid UI development (built in 3 hours)
Built-in interactivity (no JavaScript needed)
Perfect for data science demos

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

👤 Author
Michael Augustine

GitHub: @augforce
LinkedIn: https://www.linkedin.com/in/michael-augustine-8926a2164/
Website: https://linktr.ee/MichaelAugustine
For questions or collaboration: maugustine78@gmail.com

🙏 Acknowledgments
Baseball Savant for comprehensive Statcast data
Google Cloud for Vertex AI platform and credits
Streamlit for rapid prototyping framework
Tyler Glasnow for being an analytically interesting subject

⚠️ Disclaimer
This project is for educational and portfolio purposes only.

Model endpoint is not publicly accessible (prevents unauthorized cloud costs)
Predictions are for demonstration only - not for gambling or commercial use
Model accuracy varies and is subject to player evolution over time
Code is provided as reference for ML learning and portfolio evaluation
