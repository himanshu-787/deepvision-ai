.PHONY: test lint api docker

test:
	pytest

lint:
	ruff check .

api:
	uvicorn api.main:app --reload

docker:
	docker compose up --build
