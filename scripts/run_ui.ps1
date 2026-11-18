# Start the evaluator UI (Laptop 2). Pass the API address of Laptop 1.
param([string]$Api = "http://127.0.0.1:8000")
$env:PYTHONPATH = "$PWD\src"
python -m streamlit run ui\app.py --server.port 8501 -- --api $Api
