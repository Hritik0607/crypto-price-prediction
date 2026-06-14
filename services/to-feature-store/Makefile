run-dev-live:
	powershell -Command "Copy-Item -Force live.settings.env settings.env"
	uv run python run.py

run-dev-historical:
	powershell -Command "Copy-Item -Force historical.settings.env settings.env"
	uv run python run.py

run-dev-live-news:
	powershell -Command "Copy-Item -Force live.settings.news-signal.env settings.env"
	uv run python run.py

run-dev-historical-news:
	powershell -Command "Copy-Item -Force historical.settings.news-signal.env settings.env"
	uv run python run.py

build:
	docker build -f Dockerfile -t to-feature-store .

run: build
	docker run -it \
		--network redpanda_network \
		-e KAFKA_BROKER_ADDRESS=redpanda:9092 \
		to-feature-store
