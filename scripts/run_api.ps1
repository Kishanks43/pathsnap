# Start the API on this laptop (Laptop 1). Binds to 0.0.0.0 so the UI laptop can reach it.
$env:PYTHONPATH = "$PWD\src"
python -m uvicorn pathsnap.api.main:app --host 0.0.0.0 --port 8000 --log-level info
