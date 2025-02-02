FROM python:3.10-slim

USER root
RUN apt update && apt install git vim curl libgomp1 -y
RUN pip install --upgrade pip
RUN pip install fastapi uvicorn pydantic
RUN pip install pandas numpy lightgbm scikit-learn matplotlib jupyterlab
RUN pip install polars==0.19.2
RUN pip install streamlit
RUN pip install pyyaml
RUN pip install streamlit-card
COPY qiicast_backend /app/qiicast_backend
COPY qiicast_frontend /app/qiicast_frontend