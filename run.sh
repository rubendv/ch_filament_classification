#!/usr/bin/env bash

./build.sh && docker run -ti --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix martin "$@"
