# GenAI Lab Assignments

This repository contains a series of Generative AI and NLP assignments, each in its own folder.

## Structure

- Assignment 1 – GMM & Markov Chains with Prompt Engineering (Python script `code.py`)
- Assignment 2 – Prompt Engineering Techniques with Gemini (`code.py`)
- Assignment 3 – Fine‑tuning GPT‑2 for Creative Story Generation (`code.py`)
- Assignment 4 – QA Chatbot using Google Gemini (`code.py`)
- Assignment 5 – Stable Diffusion & Latent Text‑to‑Image (notebook `Assignment 5.ipynb`)
- Assignment 6 – Text‑to‑Image Generation (notebook `text_to_image.ipynb`)
- Assignment 7 – CLIP‑based Image Captioning (notebook `clip_captioning.ipynb`)
- Assignment 8 – Text‑to‑Speech Generation (notebook `text_to_speech.ipynb`)
- Assignment 9 – Video Generation from Text/Image Prompts (notebook `video_generation.ipynb`)
- Assignment10 – Multimodal Transformers & Text‑to‑Image via Cloudflare Workers AI (`code.py`, `multimodal_transformers.ipynb`)

## Requirements

- Python 3.10+
- Recommended: virtual environment per assignment
- For LLM / cloud APIs:
  - Google Gemini: `GOOGLE_API_KEY` in a `.env` file inside Assignment 2 and Assignment 4
  - Cloudflare Workers AI: `API_KEY` and `ACCOUNT_ID` in `.env` inside Assignment10

Install common dependencies (you may need more for some notebooks):

```bash
pip install -r requirements.txt  # if you create one
# or install per assignment based on imports
```

## Running the Python Assignments

From the repo root:

```bash
cd "Assignment 1"
python code.py

cd "../Assignment 2"
python code.py

cd "../Assignment 3"
python code.py

cd "../Assignment 4"
python code.py

cd "../Assignment10"
python code.py
```

Each script writes reports and results to its `output/` directory (JSON, CSV, text reports, and generated media).

## Running the Notebooks

Open the respective `.ipynb` files in VS Code or Jupyter and run all cells:

- Assignment 5/Assignment 5.ipynb
- Assignment 6/text_to_image.ipynb
- Assignment 7/clip_captioning.ipynb
- Assignment 8/text_to_speech.ipynb
- Assignment 9/video_generation.ipynb

Ensure GPU or appropriate runtime for heavy models (GPT‑2 fine‑tuning, diffusion, video generation).