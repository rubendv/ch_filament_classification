FROM ubuntu:14.04

RUN export uid=1000 gid=1000 && \
    mkdir -p /home/local && \
    echo "local:x:${uid}:${gid}:local,,,:/home/local:/bin/bash" >> /etc/passwd && \
    echo "local:x:${uid}:" >> /etc/group && \
    echo "local ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/local && \
    chmod 0440 /etc/sudoers.d/local && \
    chown ${uid}:${gid} -R /home/local
USER local
ENV HOME /home/local
WORKDIR /home/local

RUN sudo apt-get update
RUN sudo apt-get -y upgrade
RUN sudo apt-get -y install python3-pip python3-numpy python3-matplotlib python3-scipy python3-pandas libreoffice
RUN sudo pip3 install scikit-learn ipython

COPY data /home/local/data
WORKDIR /home/local/data
RUN sudo libreoffice --headless --convert-to csv /home/local/data/*.ods
RUN mkdir /home/local/results
COPY src /home/local/src
RUN sudo chown -R local.local /home/local
WORKDIR /home/local/src
CMD /bin/bash
