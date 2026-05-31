FROM tensorflow/tensorflow:1.12.0-py3

WORKDIR /workspace
COPY requirements-py36.txt /tmp/requirements-py36.txt
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r /tmp/requirements-py36.txt

COPY . /workspace
CMD ["python", "run.py", "--test", "-dd", "data/KolektorSDD"]
