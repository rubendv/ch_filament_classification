#!/usr/bin/env bash

docker build -t martin . && docker tag martin rubendv-pc:5000/martin && docker push rubendv-pc:5000/martin
