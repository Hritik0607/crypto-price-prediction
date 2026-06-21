# ── FastAPI (Python) commands ─────────────────────────────────────────────────
run:
	uv run python main.py

install:
	uv sync

# ── Test endpoints ────────────────────────────────────────────────────────────
check-health:
	curl http://127.0.0.1:8080/health

btc-prediction:
	curl "http://127.0.0.1:8080/predict?pair=BTC%2FUSD"

eth-prediction:
	curl "http://127.0.0.1:8080/predict?pair=ETH%2FUSD"

# ── Rust (deprecated) commands ────────────────────────────────────────────────
run-rust:
	cargo run
	