FROM python:3.8-slim-buster

WORKDIR /service
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /service
RUN pip install .
RUN pip list

CMD ["python", "server.py"]
