#!/bin/bash

sed 's/pytorch==/pytorch-cpu==/' requirements.txt > api/requirements.txt

docker build -t causai -f api/Dockerfile .
