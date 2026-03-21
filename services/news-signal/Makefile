run-claude:
	uv run python -m llms.claude

run-ollama:
	uv run python -m llms.ollama

########################################
## Run the service without Docker (for development)
########################################

run-dev-live:
	copy live.settings.env settings.env
	uv run python -m run

run-dev-historical:
	copy historical.settings.env settings.env
	uv run python -m run

########################################
## Build the service for Docker
########################################

build:
	docker build -f Dockerfile -t news-signal .

########################################
## Run the service with Docker
########################################

run-with-anthropic: build
	docker run -it \
		--network redpanda_network \
		-e KAFKA_BROKER_ADDRESS=redpanda:9092 \
		-e MODEL_PROVIDER=anthropic \
		--env-file anthropic_credentials.env \
		news-signal

run-with-ollama: build
	docker run -it \
		--network redpanda_network \
		-e KAFKA_BROKER_ADDRESS=redpanda:9092 \
		-e MODEL_PROVIDER=ollama \
		--env-file ollama.env \
		news-signal


########################################
## Generation of an instruction dataset with tuples (instruction, input, output)
## to do Supervised Fine Tuning
########################################

instruction-dataset-with-claude:
	uv run python golden_dataset.py \
		--model_provider anthropic \
		--n 10000 \
		--input_file ./data/cryptopanic_news.csv \
		--output_file ./data/instruction_dataset_claude_10k.jsonl

instruction-dataset-with-ollama:
	uv run python golden_dataset.py \
		--model_provider ollama \
		--n 1000 \
		--input_file ./data/cryptopanic_news.csv \
		--output_file ./data/instruction_dataset_ollama_10k.jsonl

# To install the dependencies for the GPU instance
venv-gpu-instance:
	export UV_LINK_MODE=copy && \
	export TMPDIR=/workspace/.tmp-uv && \
	export UV_INSTALLER_MAX_JOBS=1 && \
	mkdir -p /workspace/.tmp-uv && \
	curl -LsSf https://astral.sh/uv/install.sh | sh && \
	. $$HOME/.local/bin/env && \
	uv sync --no-cache --group gpu-instance

# To login to Comet ML during fine-tuning
login-comet:
	uv run comet login

# To fine-tune the model
fine-tune:
	uv run python fine_tuning.py \
		--base_llm_name unsloth/Llama-3.2-1B-bnb-4bit \
		--dataset_path ./data/instruction_dataset_ollama.jsonl \
		--comet_ml_project_name news-signal-extractor \
		--max_steps 5
