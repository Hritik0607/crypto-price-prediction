########################################
## Redpanda management
########################################

start-redpanda: 
	docker compose -f redpanda.yml up -d

stop-redpanda:
	docker compose -f redpanda.yml down

add-one-partition-to-trades-topic:
	docker compose -f redpanda.yml exec redpanda rpk topic add-partitions trades --num 1


########################################
## Technical indicators pipeline 
########################################

# for live data
build-technical-indicators-pipeline:
	docker compose -f technical-indicators-pipeline.yml build

start-technical-indicators-pipeline: build-technical-indicators-pipeline
	docker compose -f technical-indicators-pipeline.yml up -d

stop-technical-indicators-pipeline:
	docker compose -f technical-indicators-pipeline.yml down