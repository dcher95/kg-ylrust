#!/bin/bash
# Set up a link to the API key to root's home.
mkdir -p /root/.kaggle
ln -s /workspaces/kg-ylrust/kaggle.json /root/.kaggle/kaggle.json
chmod 600 /root/.kaggle/kaggle.json