FROM python:3.8-slim-buster

RUN apt-get update -y --no-install-recommends

# gcc compiler and opencv prerequisites
RUN apt-get -y --no-install-recommends install nano git build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN apt-get update && apt-get install -y --no-install-recommends \
	python3-opencv ca-certificates python3-dev git wget sudo  \
	cmake ninja-build && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt

# Install the requirements requirements
RUN pip install -r /app/requirements.txt

# Added after testing the flask code
WORKDIR /app
COPY ui.py /app/ui.py

RUN ls -la /app/

ENTRYPOINT ["streamlit", "run"]

CMD ["/app/ui.py"]