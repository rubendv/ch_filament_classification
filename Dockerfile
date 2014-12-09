FROM ubuntu:14.04

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install python3-pip python3-numpy python3-matplotlib python3-scipy python3-pandas && \
    apt-get -y install build-essential python3-dev python3-setuptools python3-numpy python3-scipy libatlas-dev libatlas3gf-base && \
    update-alternatives --set libblas.so.3 /usr/lib/atlas-base/atlas/libblas.so.3 && \
    update-alternatives --set liblapack.so.3 /usr/lib/atlas-base/atlas/liblapack.so.3 && \
    pip3 install scikit-learn ipython

RUN export uid=3652 gid=342 && \
    mkdir -p /home/local && \
    echo "local:x:${uid}:${gid}:local,,,:/home/local:/bin/bash" >> /etc/passwd && \
    echo "local:x:${uid}:" >> /etc/group && \
    echo "local ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/local && \
    chmod 0440 /etc/sudoers.d/local && \
    chown ${uid}:${gid} -R /home/local
USER local
ENV HOME /home/local
WORKDIR /home/local

COPY data /home/local/data
WORKDIR /home/local/data
RUN sudo apt-get -y install libreoffice && sudo libreoffice --headless --convert-to csv /home/local/data/*.ods && sudo apt-get -y purge libreoffice && sudo apt-get -y autoremove && sudo apt-get clean
RUN mkdir /home/local/results
COPY src /home/local/src
RUN sudo chown -R local.local /home/local
WORKDIR /home/local/src
CMD /bin/bash
