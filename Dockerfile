FROM ubuntu:14.04

RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install python3-pip python3-numpy python3-matplotlib python3-scipy python3-pandas unoconv libreoffice
RUN pip3 install scikit-learn django django-jsonfield

# Create local user and setup source code and directories
RUN useradd -mU -s /bin/bash -d /home/local local
COPY src /home/local/src
COPY data /home/local/data
RUN chown -R local.local /home/local
USER local
WORKDIR /home/local/data
RUN soffice --headless --convert-to csv *.ods
WORKDIR /home/local/src/web
RUN python3 manage.py migrate
ENV DJANGO_SETTINGS_MODULE web.settings
RUN python3 firstsetup.py
EXPOSE 8000
CMD python3 manage.py runserver 0.0.0.0:8000
