# Makefile for MES 0DTE Lotto-Grid Options Bot

.PHONY: help install test lint format run-bot run-ui docker-build docker-up docker-down clean

# Default target
help:
	@echo "MES 0DTE Lotto-Grid Options Bot - Development Commands"
	@echo ""
	@echo "Setup Commands:"
	@echo "  install         Install dependencies with Poetry"
	@echo "  install-dev     Install with development dependencies"
	@echo ""
	@echo "Development Commands:"
	@echo "  test           Run test suite"
	@echo "  test-cov       Run tests with coverage report"
	@echo "  lint           Run linting (flake8, mypy)"
	@echo "  format         Format code with black"
	@echo "  format-check   Check code formatting"
	@echo ""
	@echo "Run Commands:"
	@echo "  run-bot        Start the trading bot"
	@echo "  run-ui         Start the Streamlit dashboard"
	@echo "  run-backtest   Run a sample backtest"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build   Build Docker images"
	@echo "  docker-up      Start all services with Docker Compose"
	@echo "  docker-down    Stop all Docker services"
	@echo "  docker-logs    View Docker logs"
	@echo ""
	@echo "Utility Commands:"
	@echo "  clean          Clean temporary files and caches"
	@echo "  setup-env      Copy example environment file"
	@echo "  validate-env   Validate environment configuration"

# Setup Commands
install:
	@echo "Installing dependencies..."
	poetry install --without dev

install-dev:
	@echo "Installing development dependencies..."
	poetry install

setup-env:
	@echo "Setting up environment file..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file from .env.example"; \
		echo "Please edit .env with your configuration"; \
	else \
		echo ".env file already exists"; \
	fi

# Development Commands
test:
	@echo "Running test suite..."
	poetry run pytest -v

test-cov:
	@echo "Running tests with coverage..."
	poetry run pytest --cov=app --cov-report=html --cov-report=term tests/

lint:
	@echo "Running linting..."
	poetry run flake8 app/ tests/
	poetry run mypy app/

format:
	@echo "Formatting code..."
	poetry run black app/ tests/

format-check:
	@echo "Checking code formatting..."
	poetry run black --check app/ tests/

# Run Commands
run-bot:
	@echo "Starting trading bot..."
	@if [ ! -f .env ]; then \
		echo "Error: .env file not found. Run 'make setup-env' first."; \
		exit 1; \
	fi
	poetry run python -m app.bot

run-ui:
	@echo "Starting Streamlit dashboard..."
	@if [ ! -f .env ]; then \
		echo "Error: .env file not found. Run 'make setup-env' first."; \
		exit 1; \
	fi
	poetry run streamlit run app/ui.py --server.port 8501

run-backtest:
	@echo "Running sample backtest..."
	poetry run python -c "
import asyncio
from datetime import date, timedelta
from app.backtester import LottoGridBacktester

async def run_sample():
    backtester = LottoGridBacktester('sqlite:///./data/lotto_grid.db')
    end_date = date.today()
    start_date = end_date - timedelta(days=30)

    print(f'Running backtest from {start_date} to {end_date}...')
    results = await backtester.run_backtest(start_date, end_date, 5000.0)

    print(f'Results:')
    print(f'  Total Return: {results[\"total_return\"]:.2%}')
    print(f'  Win Rate: {results[\"win_rate\"]:.1%}')
    print(f'  Max Drawdown: \$${results[\"max_drawdown\"]:.2f}')
    print(f'  Total Trades: {results[\"total_trades\"]}')

asyncio.run(run_sample())
"

# Docker Commands
docker-build:
	@echo "Building Docker images..."
	docker-compose build

docker-up:
	@echo "Starting Docker services..."
	@if [ ! -f .env ]; then \
		echo "Error: .env file not found. Run 'make setup-env' first."; \
		exit 1; \
	fi
	docker-compose up -d

docker-down:
	@echo "Stopping Docker services..."
	docker-compose down

docker-logs:
	@echo "Viewing Docker logs..."
	docker-compose logs -f

docker-restart:
	@echo "Restarting Docker services..."
	docker-compose restart

# Validation Commands
validate-env:
	@echo "Validating environment configuration..."
	poetry run python -c "
from app.config import config
try:
    config.validate()
    print('✅ Configuration is valid')
except Exception as e:
    print(f'❌ Configuration error: {e}')
    exit(1)
"

validate-db:
	@echo "Validating database connection..."
	poetry run python -c "
from app.models import create_database
from app.config import config
try:
    create_database(config.database.url)
    print('✅ Database connection successful')
except Exception as e:
    print(f'❌ Database error: {e}')
    exit(1)
"

# Utility Commands
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name "htmlcov" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -delete

clean-docker:
	@echo "Cleaning Docker resources..."
	docker-compose down --volumes --remove-orphans
	docker system prune -f

# Data Commands
backup-data:
	@echo "Creating data backup..."
	@mkdir -p backups
	@timestamp=$$(date +%Y%m%d_%H%M%S); \
	tar -czf "backups/lotto_grid_backup_$$timestamp.tar.gz" data/ logs/ .env
	@echo "Backup created: backups/lotto_grid_backup_$$timestamp.tar.gz"

restore-data:
	@echo "To restore data, extract a backup file:"
	@echo "  tar -xzf backups/lotto_grid_backup_YYYYMMDD_HHMMSS.tar.gz"

# Development Setup
dev-setup: install-dev setup-env validate-env
	@echo "Development environment setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Edit .env with your IB credentials"
	@echo "2. Start IB Gateway/TWS with API enabled"
	@echo "3. Run 'make run-bot' to start trading"
	@echo "4. Run 'make run-ui' to access dashboard"

# Production Setup
prod-setup: install setup-env validate-env validate-db
	@echo "Production environment setup complete!"
	@echo ""
	@echo "For Docker deployment:"
	@echo "  make docker-up"
	@echo ""
	@echo "For direct deployment:"
	@echo "  make run-bot (in production)"

# CI/CD Commands
ci-test: install-dev lint test-cov
	@echo "CI/CD testing complete"

ci-build: docker-build
	@echo "CI/CD build complete"

# Health Checks
health-check:
	@echo "Performing health checks..."
	@echo "1. Checking configuration..."
	@make validate-env
	@echo "2. Checking database..."
	@make validate-db
	@echo "3. Checking Docker services (if running)..."
	@if docker-compose ps | grep -q "Up"; then \
		echo "✅ Docker services are running"; \
		docker-compose ps; \
	else \
		echo "ℹ️  Docker services not running"; \
	fi
	@echo "Health check complete!"

# Performance Testing
perf-test:
	@echo "Running performance tests..."
	poetry run python -c "
import time
import asyncio
from app.backtester import LottoGridBacktester
from datetime import date, timedelta

async def perf_test():
    backtester = LottoGridBacktester('sqlite:///:memory:')
    start_time = time.time()

    # Test with 7 days of data
    end_date = date.today()
    start_date = end_date - timedelta(days=7)

    results = await backtester.run_backtest(start_date, end_date, 5000.0)

    elapsed = time.time() - start_time
    print(f'Backtest completed in {elapsed:.2f} seconds')
    print(f'Processed {results[\"total_trades\"]} trades')

asyncio.run(perf_test())
"

# Show status
status:
	@echo "MES 0DTE Lotto-Grid Bot Status"
	@echo "=============================="
	@echo ""
	@echo "Environment:"
	@if [ -f .env ]; then \
		echo "✅ .env file exists"; \
	else \
		echo "❌ .env file missing"; \
	fi
	@echo ""
	@echo "Dependencies:"
	@if command -v poetry >/dev/null 2>&1; then \
		echo "✅ Poetry installed"; \
	else \
		echo "❌ Poetry not installed"; \
	fi
	@echo ""
	@echo "Docker:"
	@if command -v docker >/dev/null 2>&1; then \
		echo "✅ Docker installed"; \
		if docker-compose ps | grep -q "Up"; then \
			echo "✅ Services running"; \
		else \
			echo "ℹ️  Services not running"; \
		fi \
	else \
		echo "❌ Docker not installed"; \
	fi
