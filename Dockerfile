FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mon_modele_gradient_final.pkl .
COPY mon_outillage.py .
COPY app.py .
COPY interface.html .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]