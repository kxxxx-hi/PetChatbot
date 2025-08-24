# Pet Adoption Chatbot (Streamlit)

## Run locally
```bash
pip install -r requirements.txt
python -m nltk.downloader vader_lexicon
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
- Push these files to a public GitHub repo.
- In Streamlit Cloud: New App → pick repo/branch → file = `app.py` → Deploy.

## Deploy (Hugging Face Spaces)
- Create Space → Streamlit template → upload files or link repo → Hardware: CPU (free).