# Portfolio Action Plan — Xuyang Chen

## Current State Assessment

- **GitHub**: 9 repos, mostly coursework. No ML/research projects visible.
- **Website (ferriraai.com)**: Status unclear, likely outdated.
- **Gap**: Strong research credentials (IEEE paper, CHI submission, 6 DL architectures) but no public evidence of ML engineering ability.

---

## Strategy: 2 High-Impact Projects + GitHub Cleanup

### Time Budget: 10-20 hrs/week × 4 weeks = 40-80 hours total

---

## Project 1: Time Series Classification Explorer (Interactive Demo)

**Why this project**: Directly converts your existing research (6+ DL architecture comparison + Grad-CAM explainability) into a deployable, interactive tool. Shows you can turn research into product.

**What to build**: A Streamlit or Gradio web app that lets users:
1. Upload or select a time series dataset
2. Choose a model architecture (LSTM, CNN, Transformer, ResNet, etc.)
3. Run classification and see results
4. Visualize Grad-CAM attention maps showing *why* the model made its prediction
5. Compare performance across architectures with interactive charts

**Tech stack**:
- Frontend: Streamlit (fastest) or Gradio
- Models: Your existing PyTorch models (pre-trained weights)
- Visualization: Plotly for interactive charts, matplotlib for Grad-CAM heatmaps
- Deployment: Hugging Face Spaces (free hosting, GPU available)

**What makes this stand out**:
- Explainable AI (XAI) is a hot topic — Grad-CAM visualization is a strong differentiator
- Multi-architecture comparison shows breadth of knowledge
- Interactive demo means recruiters can actually *use* it, not just read about it

**Time estimate**: 20-30 hours

**Week-by-week breakdown**:

| Week | Task | Hours |
|------|------|-------|
| Week 1 | Clean up research code. Extract model definitions and training scripts into modular files. Create `requirements.txt`. | 8-10 |
| Week 2 | Build Streamlit app: dataset selection, model loading, inference pipeline, basic Grad-CAM display. | 10-12 |
| Week 3 | Add architecture comparison dashboard, polish UI, write README, deploy to Hugging Face Spaces. | 8-10 |

**GitHub repo structure**:
```
time-series-classifier/
├── README.md                    # Detailed with screenshots, live demo link
├── app.py                       # Streamlit entry point
├── models/
│   ├── lstm.py
│   ├── cnn.py
│   ├── transformer.py
│   └── resnet.py
├── explainability/
│   └── gradcam.py               # Grad-CAM implementation
├── data/
│   └── sample_datasets/         # 2-3 sample datasets
├── notebooks/
│   └── analysis.ipynb           # Research analysis notebook
├── weights/                     # Pre-trained model weights
├── requirements.txt
└── Dockerfile                   # For containerized deployment
```

**README must include**:
- Live demo link (Hugging Face Spaces)
- Screenshots/GIFs of the app in action
- Model performance comparison table
- Brief explanation of the research behind it
- Link to your IEEE paper or related publication
- Installation and usage instructions

---

## Project 2: Physics-Informed Neural Network (PINN) Visualizer

**Why this project**: You have an IEEE Xplore publication on PINNs — this is your strongest academic credential. An interactive visualizer turns a static paper into a living demo that anyone can understand.

**What to build**: An interactive web application that:
1. Shows a physics problem (e.g., heat equation, wave equation, fluid flow)
2. Lets users adjust parameters (boundary conditions, domain size, etc.)
3. Visualizes the PINN solution in real-time with animated plots
4. Compares PINN solution vs. analytical/numerical solution
5. Shows the training process: loss convergence, physics loss vs. data loss

**Tech stack**:
- Frontend: Streamlit + Plotly (3D surface plots, animations)
- Backend: PyTorch (your existing PINN implementation)
- Deployment: Hugging Face Spaces or Streamlit Cloud

**What makes this stand out**:
- Extremely rare project type — almost no one has interactive PINN demos
- Combines physics + ML in a visually compelling way
- Directly demonstrates your published research
- Shows you can communicate complex ideas to non-experts

**Time estimate**: 15-25 hours

**Week-by-week breakdown**:

| Week | Task | Hours |
|------|------|-------|
| Week 3 | Port PINN code from paper into clean, modular format. Pre-train models for 2-3 physics problems. | 8-10 |
| Week 4 | Build Streamlit app: parameter controls, real-time visualization, solution comparison. Deploy. Write README. | 10-12 |

**GitHub repo structure**:
```
pinn-visualizer/
├── README.md                    # With live demo, paper link, visuals
├── app.py                       # Streamlit entry point
├── models/
│   └── pinn.py                  # PINN architecture
├── physics/
│   ├── heat_equation.py
│   ├── wave_equation.py
│   └── burgers_equation.py
├── visualization/
│   └── plots.py                 # Plotly 3D surface plots
├── weights/                     # Pre-trained weights
├── paper/
│   └── README.md                # Link to IEEE paper + summary
├── requirements.txt
└── Dockerfile
```

---

## GitHub Profile Cleanup (Do This First — 2 hours)

Before adding new projects, clean up your existing profile:

### Immediate actions:
1. **Hide or make private** all coursework repos (ECS34-proj1, ECS34-proj4, ecs163-25s) — they add noise and make your profile look like a student, not an engineer
2. **Add a profile README** (create a repo named `1anthanum` with a `README.md`):
   - One-line bio: "ML Engineer & Researcher | Physics-Informed ML | Time Series | NLP"
   - Links to: your website, LinkedIn, IEEE paper
   - 2-3 bullet points on current research
   - Tech stack badges (PyTorch, TensorFlow, Python, etc.)
3. **Pin your 2 new project repos** once they're ready — these will be the first thing visitors see
4. **Add a professional profile photo** (same as LinkedIn)
5. **Update bio** with your affiliation: "BS Applied Physics & CS @ UC Davis | MIT MicroMasters | AI/ML Research"

### Profile README template:
```markdown
# Hi, I'm Xuyang Chen 👋

**ML Engineer & Researcher** — Applied Physics × Deep Learning

🔬 Research focus: Physics-Informed Neural Networks, Time Series Classification, NLP
🎓 UC Davis (Applied Physics & CS) | MIT MicroMasters (Statistics & Data Science)
📄 Published: IEEE Xplore (2024) — Physics-Informed Neural Networks
🔭 Currently: AI narrative systems research, CHI 2026 submission

### Featured Projects
- 🔥 [Time Series Classification Explorer](#) — Interactive multi-architecture comparison with Grad-CAM explainability
- ⚡ [PINN Visualizer](#) — Interactive physics-informed neural network demos

### Tech Stack
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)

### Connect
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/xuyang-chen-a4a8a1394/)
[![Google Scholar](https://img.shields.io/badge/Scholar-4285F4?style=flat&logo=googlescholar&logoColor=white)](#)
```

---

## Personal Website (ferriraai.com) — Update After Projects Are Done

Once both projects are deployed, update your website to include:
1. **Hero section**: Name, title, one-line pitch
2. **Projects page**: Each project with live demo link, screenshot, and description
3. **Publications**: IEEE paper link + CHI 2026 (if accepted)
4. **Resume/CV**: Downloadable PDF
5. **Contact**: Email + LinkedIn + GitHub

If your current site is outdated, the fastest approach is a clean Hugo or Next.js template. But this is lower priority than the two projects above.

---

## 4-Week Timeline Summary

| Week | Focus | Deliverable |
|------|-------|-------------|
| **Week 1** | GitHub cleanup + Time Series project: code cleanup | Clean GitHub profile, modular research code |
| **Week 2** | Time Series project: build Streamlit app | Working app with Grad-CAM visualization |
| **Week 3** | Time Series deployment + PINN project: code cleanup | Live demo on HF Spaces, clean PINN code |
| **Week 4** | PINN Streamlit app + deploy + website update | Both projects live, pinned on GitHub |

---

## What Recruiters/Hiring Managers Will See

**Before (now):**
> GitHub with coursework repos → "Just another CS student"

**After (4 weeks):**
> Clean profile with 2 polished, deployed ML projects + IEEE publication link → "This person can build AND research. They turned a published paper into a working tool."

This is the difference between getting screened out and getting an interview.

---

## Key Principles

1. **README quality matters as much as code quality.** A repo with great code but a bad README is invisible. A repo with good code and an excellent README (screenshots, live demo, clear explanation) gets starred and shared.

2. **Deploy everything.** A model that only runs locally is worth 10% of a model with a live demo link. Hugging Face Spaces is free and takes 30 minutes to set up.

3. **Don't over-scope.** Two polished projects > five half-finished ones. If you find yourself spending more than the allocated time, cut features, not quality.

4. **Connect projects to your narrative.** Every project should reinforce the story: "I'm an ML engineer with deep research background who can turn complex ideas into usable tools."
