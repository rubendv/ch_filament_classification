#!/usr/bin/env bash

docker pull rubendv-pc:5000/martin && docker run -ti --rm -v $PWD/results:/home/local/results:rw rubendv-pc:5000/martin "$@"
