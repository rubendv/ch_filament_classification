#!/usr/bin/env bash

docker build -t martin . && docker tag -f martin rubendv-pc:5000/martin && docker push rubendv-pc:5000/martin
