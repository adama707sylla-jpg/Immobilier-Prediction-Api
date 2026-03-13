FROM python:3.11-slim 

WORKDIR /app

COPY mon_modele_gradient_final.pkl .
COPY mon_outillage.py .
COPY app.py .

RUN pip install joblib scikit-learn numpy pandas

CMD ["python", "app.py"]