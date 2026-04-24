.PHONY: help install up down dev run check reset clean

COMPOSE := docker compose -f ../frontend/docker-compose.yml

help:
	@echo "Common:"
	@echo "  make install    Install Python deps via uv"
	@echo "  make up         Start MinIO + MongoDB (docker compose)"
	@echo "  make down       Stop MinIO + MongoDB"
	@echo "  make dev        Run API with hot reload on :8000"
	@echo "  make run        Run API on :8000 (all interfaces, LAN-reachable)"
	@echo "  make check      Hit / and /reports"
	@echo "  make reset      Wipe reports collection + MinIO bucket"
	@echo "  make clean      Remove __pycache__"

install:
	uv sync

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

dev:
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

run:
	uv run uvicorn src.main:app --host 0.0.0.0 --port 8000

check:
	@echo "--- / ---"
	@curl -s http://127.0.0.1:8000/ | python3 -m json.tool
	@echo "\n--- /reports ---"
	@curl -s http://127.0.0.1:8000/reports | python3 -m json.tool

reset:
	@echo "Wiping dammage.reports + MinIO dammage/*"
	@docker exec dammage-mongo mongosh --quiet --eval 'db.getSiblingDB("dammage").reports.deleteMany({})' || true
	@docker exec dammage-minio mc alias set local http://127.0.0.1:9000 admin password123 >/dev/null 2>&1 || true
	@docker exec dammage-minio mc rm --recursive --force local/dammage/ >/dev/null 2>&1 || true

clean:
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
