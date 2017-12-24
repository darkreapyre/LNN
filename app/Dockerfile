FROM python:3.6

# Install build-essential, git, wget and other dependencies
RUN apt-get update && apt-get install -y \
  build-essential \
  git \
  libopenblas-dev \
  libatlas-base-dev \
  gfortran \
  gcc \
#  libopencv-dev \
#  libav-tools \
#  python3-dev \
#  python-setuptools \
  wget \
#  python3-pip \
  nginx \
  supervisor && \
  rm -rf /var/lib/apt/lists/*

#install Flask
RUN pip3 install --upgrade pip
RUN pip3 install \
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
  requests


ADD ./src /app
ADD ./config /config

#RUN pip3 install -r /config/requirements.txt


#RUN echo "\ndaemon off;" >> /etc/nginx/nginx.conf
#RUN rm /etc/nginx/sites-enabled/default

#RUN ln -s /config/nginx.conf /etc/nginx/sites-enabled/
#RUN ln -s /config/supervisor.conf /etc/supervisor/conf.d/

EXPOSE 5000

#CMD ["supervisord", "-n"]
CMD ["python", "app/app.py"]
