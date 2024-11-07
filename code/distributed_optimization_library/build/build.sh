#!/bin/bash
docker buildx build -t k3nfalt/python_ml --platform linux/amd64 --push .