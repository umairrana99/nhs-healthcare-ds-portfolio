FROM python:3.12-slim

WORKDIR /app

# Install the package (core dependencies only — no dev/test tooling).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

# The trained model is provided at runtime (mounted volume — see docker-compose.yml).
ENV READMISSION_MODEL_PATH=/app/models/model.joblib

EXPOSE 8000
CMD ["uvicorn", "readmission.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
