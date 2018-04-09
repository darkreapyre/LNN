# Copyright 2017-2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Amazon Software License (the "License"). You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#    http://aws.amazon.com/asl/
#
# or in the "license" file accompanying this file.
# This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#

FROM ubuntu:14.04.5

ARG ID
ENV BUILD_ID=$ID
RUN echo $BUILD_ID

#ENV LANG="C.UTF-8"

# Building git from source code:
#   Ubuntu's default git package is built with broken gnutls. Rebuild git with openssl.
##########################################################################
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        wget=1.15-* fakeroot=1.20-* ca-certificates \
    && apt-get -qy build-dep git=1:1.9.1 \
    && apt-get -qy install libcurl4-openssl-dev=7.35.0-* git-man=1:1.9.1-* liberror-perl=0.17-* \
    && mkdir -p /usr/src/git-openssl \
    && cd /usr/src/git-openssl \
    && apt-get source git=1:1.9.1 \
    && cd $(find -mindepth 1 -maxdepth 1 -type d -name "git-*") \
    && sed -i -- 's/libcurl4-gnutls-dev/libcurl4-openssl-dev/' ./debian/control \
    && sed -i -- '/TEST\s*=\s*test/d' ./debian/rules \
    && dpkg-buildpackage -rfakeroot -b \
    && find .. -type f -name "git_*ubuntu*.deb" -exec dpkg -i \{\} \; \
    && rm -rf /usr/src/git-openssl \
    && rm -rf /var/lib/apt/lists/* \
# Install dependencies by all python images equivalent to buildpack-deps:jessie
# on the public repos.
    && apt-get update && apt-get install -y --no-install-recommends autoconf=2.69-* automake=1:1.14.1-* \
        bzip2=1.0.6-* file=1:5.14-* g++=4:4.8.2-* gcc=4:4.8.2-* imagemagick=8:6.7.7.10-* \
        libbz2-dev=1.0.6-* libc6-dev=2.19-* libcurl4-openssl-dev=7.35.0-* curl=7.35.0-* \
        libdb-dev=1:5.3.21~* libevent-dev=2.0.21-stable-* libffi-dev=3.1~rc1+r3.0.13-* \
        libgeoip-dev=1.6.0-* libglib2.0-dev=2.40.2-* libjpeg-dev=8c-* \
        libkrb5-dev=1.12+dfsg-* liblzma-dev=5.1.1alpha+20120614-* libmagickcore-dev=8:6.7.7.10-* \
        libmagickwand-dev=8:6.7.7.10-* libmysqlclient-dev=5.5.59-* libncurses5-dev=5.9+20140118-* \
        libpng12-dev=1.2.50-* libpq-dev=9.3.22-* libreadline-dev=6.3-* libsqlite3-dev=3.8.2-* \
        libssl-dev=1.0.1f-* libtool=2.4.2-* libwebp-dev=0.4.0-* libxml2-dev=2.9.1+dfsg1-* \
        libxslt1-dev=1.1.28-* libyaml-dev=0.1.4-* make=3.81-* patch=2.7.1-* xz-utils=5.1.1alpha+20120614-* \
        zlib1g-dev=1:1.2.8.dfsg-* tcl=8.6.0+* tk=8.6.0+* ca-certificates \
        e2fsprogs=1.42.9-* iptables=1.4.21-* xfsprogs=3.1.9ubuntu2 xz-utils=5.1.1alpha+20120614-* \
    && apt-get install -y -qq less=458-* groff=1.22.2-* \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/usr/local/bin:$PATH" \
    GPG_KEY="C01E1CAD5EA2C4F0B8E3571504C367C218ADD4FF" \
    PYTHON_VERSION="2.7.12" \
    PYTHON_PIP_VERSION="8.1.2"

RUN apt-get update && apt-get install -y --no-install-recommends \
    tcl-dev tk-dev \
    && rm -rf /var/lib/apt/lists/* \
	&& wget -O python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz" \
	&& wget -O python.tar.xz.asc "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz.asc" \
	&& export GNUPGHOME="$(mktemp -d)" \
	&& (gpg --keyserver ha.pool.sks-keyservers.net --recv-keys "$GPG_KEY" \
        || gpg --keyserver pgp.mit.edu --recv-keys "$GPG_KEY" \
        || gpg --keyserver keyserver.ubuntu.com --recv-keys "$GPG_KEY") \
	&& gpg --batch --verify python.tar.xz.asc python.tar.xz \
	&& rm -r "$GNUPGHOME" python.tar.xz.asc \
	&& mkdir -p /usr/src/python \
	&& tar -xJC /usr/src/python --strip-components=1 -f python.tar.xz \
	&& rm python.tar.xz \
	\
	&& cd /usr/src/python \
	&& ./configure \
		--enable-shared \
		--enable-unicode=ucs4 \
	&& make -j$(nproc) \
	&& make install \
	&& ldconfig \
		&& wget -O /tmp/get-pip.py 'https://bootstrap.pypa.io/get-pip.py' \
		&& python2 /tmp/get-pip.py "pip==$PYTHON_PIP_VERSION" \
		&& rm /tmp/get-pip.py \
# we use "--force-reinstall" for the case where the version of pip we're trying to install is the same as the version bundled with Python
# ("Requirement already up-to-date: pip==8.1.2 in /usr/local/lib/python3.6/site-packages")
# https://github.com/docker-library/python/pull/143#issuecomment-241032683
	&& pip install --no-cache-dir --upgrade --force-reinstall "pip==$PYTHON_PIP_VERSION" \
        && pip install awscli --no-cache-dir \
        && pip install --no-cache-dir \ 
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

# then we use "pip list" to ensure we don't have more than one pip version installed
# https://github.com/docker-library/python/pull/100
	&& [ "$(pip list |tac|tac| awk -F '[ ()]+' '$1 == "pip" { print $2; exit }')" = "$PYTHON_PIP_VERSION" ] \
	&& find /usr/local -depth \
		\( \
			\( -type d -a -name test -o -name tests \) \
			-o \
			\( -type f -a -name '*.pyc' -o -name '*.pyo' \) \
		\) -exec rm -rf '{}' + \
	&& apt-get purge -y --auto-remove tcl-dev tk-dev \
    && rm -rf /usr/src/python ~/.cache

# Configure API Endpoint
ADD ./src /app
ADD ./src/config /config
EXPOSE 80

# Launch Flask App
CMD ["python", "app/app.py"]