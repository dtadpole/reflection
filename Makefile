.PHONY: install lint lint-fix test test-unit test-service test-integration test-verifier test-retriever test-kb

install:
	uv sync

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check . --fix

test:
	uv run pytest

test-unit:
	uv run pytest tests/unit/ -v

# Requires SSH access to _one/_two and tunnels: reflection services tunnel start
test-service: test-unit
	uv run pytest tests/integration/test_services.py -v -s

test-integration: test-service
	uv run pytest tests/integration/ -v

# Requires tunnels: reflection services tunnel start
test-verifier:
	uv run pytest tests/integration/test_verifier_tool.py -v -s

test-retriever:
	uv run pytest tests/integration/test_retriever_tool.py -v -s

test-kb:
	uv run pytest tests/integration/test_knowledge_base.py -v -s
