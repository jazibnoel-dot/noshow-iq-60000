FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
RUN adduser --disabled-password --no-create-home appuser
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .
RUN mkdir -p models && python train.py
USER appuser
EXPOSE 7860
CMD ["uvicorn", "noshow_iq.api:app", "--host", "0.0.0.0", "--port", "7860"]