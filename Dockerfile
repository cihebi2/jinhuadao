FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt \
    && useradd -m appuser \
    && chown -R appuser /app
USER appuser
COPY app ./app
EXPOSE 8501
CMD ["streamlit", "run", "app/ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
