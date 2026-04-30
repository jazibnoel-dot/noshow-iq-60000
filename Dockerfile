FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN grep -v "^-e" requirements.txt > requirements_external.txt && pip install --no-cache-dir -r requirements_external.txt

FROM python:3.11-slim
WORKDIR /app
RUN adduser --disabled-password --no-create-home appuser
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .
RUN pip install --no-cache-dir -e .
RUN mkdir -p models 
USER appuser
EXPOSE 7860
CMD ["uvicorn", "noshow_iq.api:app", "--host", "0.0.0.0", "--port", "7860"]
RUN apt-get install -y git git-lfs && git lfs install && git lfs pull