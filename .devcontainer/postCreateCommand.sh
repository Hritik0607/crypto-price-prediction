#!/bin/bash

# Install tools specified in mise.toml
#
# cd /workspaces/real-time-ml-system-cohort-4
cd /workspaces/crypto-price-predictor
mise trust
mise install
echo 'eval "$(/usr/local/bin/mise activate bash)"' >> ~/.bashrc
source ~/.bashrc
