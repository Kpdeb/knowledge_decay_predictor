FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p models/saved
EXPOSE 8001
CMD ["sh", "-c", "python training/train.py && uvicorn main:app --host 0.0.0.0 --port 8001"]
