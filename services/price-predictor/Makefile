reset-venv:
	uv clean
	rm -rf .venv
	rm uv.lock
	uv sync

training:
	uv run python training.py

inference:
	uv run python inference.py

build:
	docker build -f Dockerfile -t price-predictor .

run-training: build
	# Run the dockerized trainining job 6
	docker run -it \
		-e TRAINING_OR_INFERNCE=training \
		price-predictor

run-inference: build
	# Run the dockerized inference job
	docker run -it \
		-e TRAINING_OR_INFERNCE=inference \
		--network redpanda_network \
		-e KAFKA_BROKER_ADDRESS=redpanda:9092 \
		-e ELASTICSEARCH_URL=http://elasticsearch:9200 \
		price-predictor
