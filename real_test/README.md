# Real-file test

`files/` holds 40 real files (no personal documents): lecture PDFs, notes, a CSV,
diagrams and charts with text, 2 Korean notices, 3 photos with no text and 1 exact
duplicate. `test_questions.csv` has 32 questions with the correct file for each.

## Run

```bash
RECALL_HOME=~/.recall_real .venv/bin/python -m recall index real_test/files
RECALL_HOME=~/.recall_real .venv/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8502
```

Open http://localhost:8502. A separate `RECALL_HOME` keeps this index apart from the demo index.

## Score

Search each question in Agent mode and write where the correct file appears
(1, 2, 3 ... or none) in the last column of `test_questions.csv`.

- Hit@1 = questions with rank 1 / number of questions
- Recall@5 = questions with rank 1 to 5 / number of questions

Report the 29 normal questions and the 3 "expected to fail" ones separately.
Korean text and photos with no text are known limits (English OCR only, no image search).
