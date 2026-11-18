FROM python:3.13-slim

WORKDIR /app
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY artifacts ./artifacts

# The student/evaluator trip folders are mounted at run time, never baked into the image:
#   docker run -p 8000:8000 -v "%cd%/data/raw:/app/data/raw" pathsnap-api
EXPOSE 8000
CMD ["uvicorn", "pathsnap.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
