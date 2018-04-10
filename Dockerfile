FROM python:3.6
ARG ID
ENV BUILD_ID=$ID
RUN echo $BUILD_ID

# Install build-essential, git, wget and other dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  git \
  libopenblas-dev \
  libatlas-base-dev \
  gfortran \
  gcc \
  wget \
  curl \
  nginx \
  supervisor && \
  rm -rf /var/lib/apt/lists/*

# Install Flask API Libraries
RUN pip3 install --upgrade pip
RUN pip3 install \
  uwsgi \
  Flask \
  matplotlib \
  scikit-image \
  Pillow \
  Jinja2 \
  Werkzeug \
  certifi \
  gunicorn \
  sagemaker \
  mxnet

# Resolve issues with python-dateutil and botocore
RUN pip install "python-dateutil<2.7.0,>=2.1"

# Configure API Endpoint
ADD ./src /app
ADD ./src/config /config
EXPOSE 80

# Launch Flask App
CMD ["python", "app/app.py"]