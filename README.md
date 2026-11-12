# TechPulse Skill Academy — Company Knowledge Assistant

Module 08: Enterprise RAG Assignment. This is a fictional classroom prototype.

## Files
- `app.py` — Streamlit application and retrieval experiments
- `documents/` — five Markdown knowledge-base documents
- `tests/test_questions.csv` — 20 required test cases; fill actual-result columns after running
- `report/project_report_template.md` — report template; add measured findings
- `.env.example` — placeholder only
- `requirements.txt` — Python dependencies

## Windows 11 setup
1. Install Python 3.11 or a compatible Python version.
2. Extract this ZIP.
3. Open the extracted `TechPulse-RAG-Markdown` folder in VS Code.
4. Open VS Code Terminal (Terminal → New Terminal).
5. Create virtual environment:
   `py -m venv .venv`
6. Activate it:
   `.venv\Scripts\activate`
7. Install packages:
   `python -m pip install --upgrade pip`
   `pip install -r requirements.txt`
8. Copy `.env.example` and rename the copy to `.env`.
9. Open `.env` and replace the placeholder with your own Groq API key:
   `GROQ_API_KEY=your_actual_groq_api_key`
10. Run:
    `streamlit run app.py`

## How to use
- Ask a normal question in the Ask TechPulse tab.
- Open Sources and supporting evidence to inspect filenames and retrieved text.
- For an old-policy question, explicitly say "old policy" or "version 1.0".
- Current-rule questions use the current policy.
- In Retrieval experiments, run the six fixed questions for k=1, 3, and 5.
- Change chunk size/overlap in the sidebar and repeat the three chunk experiment questions.
- Download experiment output and evaluate evidence relevance and answer correctness manually.
- Fill all actual-answer, actual-source, pass/fail, and reason columns in `tests/test_questions.csv`.

## Important
- All company names, contact details, fees, dates, and policies are fictional.
- Never submit `.env`, real API keys, `.venv`, downloaded embedding models, or local cache folders.
- The app rebuilds Chroma from the supplied Markdown files in memory.
- The Groq model ID is configured as `openai/gpt-oss-120b`; if unavailable, check Groq's current model catalogue.
- Prompt-injection protection is a mitigation, not a guarantee. Test and report actual behavior.
- Do not invent test results or claim the application is production-ready.
