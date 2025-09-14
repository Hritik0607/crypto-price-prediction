#!/bin/bash

echo "Creating namespace named 'metabase' if it doesn't exist..."
kubectl create namespace metabase --dry-run=client -o yaml | kubectl apply -f -


echo "Applying Metabase manifests..."
kubectl apply -f manifests/metabase-all-in-one.yaml

# Display connection information
echo "Metabase is being deployed. Once ready, you can access it via port-forwarding to localhost"