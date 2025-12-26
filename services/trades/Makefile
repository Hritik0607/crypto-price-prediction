run-dev-live:
	cp live.settings.env settings.env
	uv run python run.py

run-dev-historical:
	powershell -Command "Copy-Item -Force historical.settings.env settings.env"
	uv run python run.py

build:
	docker build -f Dockerfile -t trades .

run: build
	docker run -it \
		--network redpanda_network \
		-e KAFKA_BROKER_ADDRESS=redpanda:9092 \
		trades

build-multistage:
	docker build -f multistage.Dockerfile -t trades:multistage .

run-multistage: build-multistage
	docker run -it \
		--network redpanda_network \
		-e KAFKA_BROKER_ADDRESS=redpanda:9092 \
		trades:multistage

build-all: build build-multistage