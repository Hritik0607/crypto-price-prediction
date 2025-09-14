# # Runs the trades service as a standalone Pyton app (not Dockerized)
# dev:
# 	uv run services/${service}/src/${service}/main.py

# # Builds and pushes the docker image to the given environment
# build-and-push:
# 	./scripts/build-and-push-image.sh ${image} ${env}

# # Deploys a service to the given environment
# deploy:
# 	./scripts/deploy.sh ${service} ${env}

# lint:
# 	ruff check . --fix

dev:
	uv run services/${service}/src/${service}/main.py

build:
	docker build -t ${service}:dev -f docker/${service}.Dockerfile .

push:
	kind load docker-image ${service}:dev --name rwml-34fa

deploy: build push
	kubectl delete -f deployments/dev/${service}/${service}.yaml --ignore-not-found=true
	kubectl apply -f deployments/dev/${service}/${service}.yaml

lint:
	ruff check . --fix