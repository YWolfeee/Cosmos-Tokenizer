#!/bin/bash

apt-get update
apt-get install -y ffmpeg git-lfs
git lfs pull
pip3 install -e .
