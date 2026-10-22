FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.1.3 scikit-learn==1.5.2 boto3==1.35.54
WORKDIR /app
COPY securefl /app/securefl
COPY cloud_node.py /app/
ENTRYPOINT ["python", "/app/cloud_node.py"]
