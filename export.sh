#!/usr/bin/bash

./build.sh

docker save fighttumorda5s | gzip -c > fighttumorda5s.tar.gz
