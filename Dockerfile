FROM ubuntu:16.04

ARG ID
ENV BUILD_ID=$ID
RUN echo $BUILD_ID

RUN apt-get update && apt-get -y install \
    build-essential \
    libopencv-dev \
    libopenblas-dev \
    libjemalloc-dev \
    libgfortran3 \
    python-dev \
    python3-dev \
    git \
    wget \
    curl \
    nginx \
    supervisor && \
    rm -rf /var/lib/apt/lists/*

# Symlink /usr/bin/python to the python 2.
RUN rm /usr/bin/python && ln -s "/usr/bin/python3" /usr/bin/python

# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    /usr/bin/python get-pip.py

RUN cd /tmp && \
    git clone --recursive https://github.com/apache/incubator-mxnet mxnet && \
    cd /tmp/mxnet && \
    git checkout tags/1.1.0 -b 1.1.0 && git submodule update --init --recursive && \
    make -j$(nproc) USE_BLAS=openblas USE_MKL2017=1 USE_DIST_KVSTORE=1 && \
    cd /tmp/mxnet/python && \
    python setup.py install && \
    cd / && \
    rm -fr /tmp/mxnet

# https://stackoverflow.com/questions/29274638/opencv-libdc1394-error-failed-to-initialize-libdc1394
RUN ln -s /dev/null /dev/raw1394

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib"

RUN pip install \
    uwsgi \
    Flask \
    numpy \
    matplotlib \
    scipy \
    scikit-image \
    Pillow \
    boto3 \
    Jinja2 \
    Werkzeug \
    certifi \
    gunicorn\
    requests \
    h5py \
    urllib3 \
    sagemaker

ADD ./src /app
ADD ./src/config /config
EXPOSE 80

CMD ["python", "app/app.py"]