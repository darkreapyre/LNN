FROM scratch
ADD amzn-container-minimal-2017.03.1.20170812-x86_64.tar.xz /
RUN mkdir /usr/src/srpm && \
    curl -o /usr/src/srpm/srpm-bundle.tar.gz "https://amazon-linux-docker-sources.s3-accelerate.amazonaws.com/srpm-bundle.tar.gz?versionId=ZVTePIIr6bdWA56CVZ7EkRKV5So3o4hl" && \
    echo "9be0315c2e647f7d1ba59fd843e5ea7bc4506028fa1329415477618a42aaa4b2 /usr/src/srpm/srpm-bundle.tar.gz" | sha256sum -c -

RUN yum install -y gcc zlib zlib-devel openssl openssl-devel curl bzip2 git zip \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
#    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local/ \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3.6.1 \
    && conda update conda \
    && conda clean --all --yes \
#    && rpm -e --nodeps curl bzip2 \
    && yum clean all

RUN conda install -y jupyter_client jupyter_console jupyter_core notebook

RUN pip --no-cache-dir install \
        awscli \
        virtualenv \
        boto3 \
        numpy \
        matplotlib \
        scipy \
        scikit-learn \
        scikit-image \
        docopt \
        schema \
        path.py \
        addict \
        requests \
        bottle \
        tornado \
        ptvsd==3.0.0 \
        h5py \
        Pillow \
        redis

ENV LD_LIBRARY_PATH /usr/local/lib

RUN jupyter notebook --generate-config --allow-root && \
    echo "c.NotebookApp.password = u'sha1:eb9f67ab8656:cb5ee4b6bfd15a0fa315e395e37585283cf985e7'" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.notebook_dir='/root/'" >> ~/.jupyter/jupyter_notebook_config.py

ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip='*'", "--allow-root"]
 

#CMD ["/bin/bash"]