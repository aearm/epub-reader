# EPUB Reader Deployment Makefile
# Usage: make <target>

.PHONY: help setup init plan apply deploy deploy-backend deploy-frontend deploy-cloud-worker destroy ssh clean watch-jobs worker-up worker-down worker-logs worker-status

# Variables
TERRAFORM_DIR = terraform
BACKEND_DIR = backend
FRONTEND_DIR = frontend
SSH_KEY_PATH = ~/.ssh/epub_reader
WORKER_CONTAINERS ?= 1

# Load local environment overrides from root .env when present.
ifneq (,$(wildcard .env))
include .env
export
endif

# Colors
GREEN = \033[0;32m
YELLOW = \033[1;33m
NC = \033[0m # No Color

help: ## Show this help
	@echo "EPUB Reader Deployment"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# =============================================================================
# Setup
# =============================================================================

setup: ssh-key ## Initial setup (create SSH key, install dependencies)
	@echo "$(GREEN)Setup complete!$(NC)"

ssh-key: ## Generate SSH key pair for EC2
	@if [ ! -f $(SSH_KEY_PATH) ]; then \
		echo "$(YELLOW)Generating SSH key...$(NC)"; \
		ssh-keygen -t rsa -b 4096 -f $(SSH_KEY_PATH) -N "" -C "epub-reader"; \
		echo "$(GREEN)SSH key created at $(SSH_KEY_PATH)$(NC)"; \
	else \
		echo "SSH key already exists at $(SSH_KEY_PATH)"; \
	fi

# =============================================================================
# Terraform
# =============================================================================

init: ## Initialize Terraform
	@echo "$(YELLOW)Initializing Terraform...$(NC)"
	cd $(TERRAFORM_DIR) && terraform init

plan: ## Plan Terraform changes
	@echo "$(YELLOW)Planning Terraform changes...$(NC)"
	cd $(TERRAFORM_DIR) && terraform plan

apply: ## Apply Terraform changes
	@echo "$(YELLOW)Applying Terraform changes...$(NC)"
	cd $(TERRAFORM_DIR) && terraform apply

destroy: ## Destroy all infrastructure (DANGEROUS!)
	@echo "$(YELLOW)WARNING: This will destroy all infrastructure!$(NC)"
	@read -p "Are you sure? Type 'yes' to confirm: " confirm && [ "$$confirm" = "yes" ]
	cd $(TERRAFORM_DIR) && terraform destroy

outputs: ## Show Terraform outputs
	cd $(TERRAFORM_DIR) && terraform output

# =============================================================================
# Deployment
# =============================================================================

deploy: generate-config deploy-backend deploy-frontend deploy-cloud-worker ## Deploy everything

generate-config: ## Generate frontend config from Terraform outputs
	@echo "$(YELLOW)Generating frontend config...$(NC)"
	$(eval API_URL := $(shell cd $(TERRAFORM_DIR) && terraform output -raw api_url 2>/dev/null))
	$(eval COGNITO_POOL_ID := $(shell cd $(TERRAFORM_DIR) && terraform output -raw cognito_user_pool_id 2>/dev/null))
	$(eval COGNITO_CLIENT_ID := $(shell cd $(TERRAFORM_DIR) && terraform output -raw cognito_client_id 2>/dev/null))
	$(eval COGNITO_DOMAIN := $(shell cd $(TERRAFORM_DIR) && terraform output -raw cognito_domain 2>/dev/null))
	$(eval AUDIO_BUCKET_URL := $(shell cd $(TERRAFORM_DIR) && terraform output -raw audio_bucket_url 2>/dev/null))
	$(eval AWS_REGION := $(shell cd $(TERRAFORM_DIR) && terraform output -raw aws_region 2>/dev/null || echo "us-east-2"))
	@sed -e 's|$${API_URL}|$(API_URL)|g' \
		-e 's|$${COGNITO_USER_POOL_ID}|$(COGNITO_POOL_ID)|g' \
		-e 's|$${COGNITO_CLIENT_ID}|$(COGNITO_CLIENT_ID)|g' \
		-e 's|$${COGNITO_DOMAIN}|$(COGNITO_DOMAIN)|g' \
		-e 's|$${AUDIO_BUCKET_URL}|$(AUDIO_BUCKET_URL)|g' \
		-e 's|$${AWS_REGION}|$(AWS_REGION)|g' \
		$(FRONTEND_DIR)/js/config.js.template > $(FRONTEND_DIR)/js/config.js
	@echo "$(GREEN)Config generated at $(FRONTEND_DIR)/js/config.js$(NC)"

deploy-backend: ## Deploy backend to EC2
	@echo "$(YELLOW)Deploying backend...$(NC)"
	$(eval EC2_IP := $(shell cd $(TERRAFORM_DIR) && terraform output -raw ec2_public_ip 2>/dev/null))
	$(eval AUDIO_BUCKET := $(shell cd $(TERRAFORM_DIR) && terraform output -raw audio_bucket 2>/dev/null))
	$(eval AWS_REGION := $(shell cd $(TERRAFORM_DIR) && terraform output -raw aws_region 2>/dev/null || echo "us-east-2"))
	$(eval COGNITO_POOL_ID := $(shell cd $(TERRAFORM_DIR) && terraform output -raw cognito_user_pool_id 2>/dev/null))
	$(eval COGNITO_CLIENT_ID := $(shell cd $(TERRAFORM_DIR) && terraform output -raw cognito_client_id 2>/dev/null))
	$(eval AUDIO_SQS_QUEUE_URL := $(shell cd $(TERRAFORM_DIR) && terraform output -raw audio_sqs_queue_url 2>/dev/null || echo ""))
	$(eval AUDIO_SQS_REGION := $(shell cd $(TERRAFORM_DIR) && terraform output -raw audio_sqs_region 2>/dev/null || echo "$(AWS_REGION)"))
	$(eval AUDIO_SQS_DLQ_URL := $(shell cd $(TERRAFORM_DIR) && terraform output -raw audio_sqs_dlq_url 2>/dev/null || echo ""))
	$(eval AUDIO_SQS_WAIT_SECONDS := $(shell cd $(TERRAFORM_DIR) && terraform output -raw audio_sqs_wait_seconds 2>/dev/null || echo "2"))
	$(eval AUDIO_SQS_VISIBILITY_TIMEOUT := $(shell cd $(TERRAFORM_DIR) && terraform output -raw audio_sqs_visibility_timeout 2>/dev/null || echo "180"))
	$(eval AUDIO_SQS_MAX_RECEIVE_COUNT := $(shell cd $(TERRAFORM_DIR) && terraform output -raw audio_sqs_max_receive_count 2>/dev/null || echo "8"))
	$(eval TF_WORKER_SHARED_SECRET := $(shell cd $(TERRAFORM_DIR) && terraform output -raw worker_shared_secret 2>/dev/null || echo ""))
	$(eval WORKER_SHARED_SECRET_VALUE := $(if $(strip $(TF_WORKER_SHARED_SECRET)),$(TF_WORKER_SHARED_SECRET),$(WORKER_SHARED_SECRET)))
	$(eval WORKER_SHARED_SECRET_SECRET_ID_VALUE := $(WORKER_SHARED_SECRET_SECRET_ID))
	$(eval WORKER_SHARED_SECRET_PARAMETER_NAME_VALUE := $(if $(strip $(WORKER_SHARED_SECRET_PARAMETER_NAME)),$(WORKER_SHARED_SECRET_PARAMETER_NAME),/epub-reader/worker-shared-secret))
	@if [ -z "$(EC2_IP)" ]; then \
		echo "Error: Could not get EC2 IP. Run 'make apply' first."; \
		exit 1; \
	fi
	@echo "Setting up directory permissions..."
	ssh -i $(SSH_KEY_PATH) -o StrictHostKeyChecking=no ec2-user@$(EC2_IP) \
		"sudo mkdir -p /opt/epub-reader && sudo chown -R ec2-user:ec2-user /opt/epub-reader"
	@echo "Copying files to EC2 ($(EC2_IP))..."
	scp -i $(SSH_KEY_PATH) \
		$(BACKEND_DIR)/coordinator.py \
		$(BACKEND_DIR)/requirements.txt \
		ec2-user@$(EC2_IP):/opt/epub-reader/
	@echo "Updating coordinator environment..."
	ssh -i $(SSH_KEY_PATH) ec2-user@$(EC2_IP) \
		"printf '%s\n' \
			'AUDIO_BUCKET=$(AUDIO_BUCKET)' \
			'AWS_REGION=$(AWS_REGION)' \
			'COGNITO_POOL_ID=$(COGNITO_POOL_ID)' \
			'COGNITO_CLIENT_ID=$(COGNITO_CLIENT_ID)' \
			'AUDIO_SQS_QUEUE_URL=$(AUDIO_SQS_QUEUE_URL)' \
			'AUDIO_SQS_REGION=$(AUDIO_SQS_REGION)' \
			'AUDIO_SQS_DLQ_URL=$(AUDIO_SQS_DLQ_URL)' \
			'AUDIO_SQS_WAIT_SECONDS=$(AUDIO_SQS_WAIT_SECONDS)' \
			'AUDIO_SQS_VISIBILITY_TIMEOUT=$(AUDIO_SQS_VISIBILITY_TIMEOUT)' \
			'AUDIO_SQS_MAX_RECEIVE_COUNT=$(AUDIO_SQS_MAX_RECEIVE_COUNT)' \
			'WORKER_SHARED_SECRET=$(WORKER_SHARED_SECRET_VALUE)' \
			'WORKER_SHARED_SECRET_SECRET_ID=$(WORKER_SHARED_SECRET_SECRET_ID_VALUE)' \
			'WORKER_SHARED_SECRET_PARAMETER_NAME=$(WORKER_SHARED_SECRET_PARAMETER_NAME_VALUE)' \
			'OPENAI_API_KEY=$(OPENAI_API_KEY)' \
			'OPENAI_MODEL=$(OPENAI_MODEL)' \
			'OPENAI_CHAT_MODEL=$(OPENAI_CHAT_MODEL)' \
			'OPENAI_TRANSLATION_MODEL=$(OPENAI_TRANSLATION_MODEL)' \
			'OPENAI_API_BASE=$(OPENAI_API_BASE)' \
			> /opt/epub-reader/.env"
	@echo "Setting up Python environment..."
	ssh -i $(SSH_KEY_PATH) ec2-user@$(EC2_IP) \
		"cd /opt/epub-reader && python3 -m venv venv 2>/dev/null || true && ./venv/bin/python -m pip install -r requirements.txt"
	@echo "Restarting coordinator service..."
	ssh -i $(SSH_KEY_PATH) ec2-user@$(EC2_IP) \
		"sudo systemctl restart coordinator || echo 'Service not configured yet'"
	@echo "$(GREEN)Backend deployed!$(NC)"

deploy-cloud-worker: ## Build/push cloud worker image and roll ECS service
	@echo "$(YELLOW)Deploying cloud worker...$(NC)"
	$(eval WORKER_REGION := $(shell cd $(TERRAFORM_DIR) && terraform output -raw audio_sqs_region 2>/dev/null || echo "us-east-2"))
	$(eval ECR_REPO := $(shell cd $(TERRAFORM_DIR) && terraform output -raw cloud_worker_ecr_repository_url 2>/dev/null))
	$(eval ECS_CLUSTER := $(shell cd $(TERRAFORM_DIR) && terraform output -raw cloud_worker_ecs_cluster_name 2>/dev/null))
	$(eval ECS_SERVICE := $(shell cd $(TERRAFORM_DIR) && terraform output -raw cloud_worker_ecs_service_name 2>/dev/null))
	@if [ -z "$(ECR_REPO)" ] || [ -z "$(ECS_CLUSTER)" ] || [ -z "$(ECS_SERVICE)" ]; then \
		echo "Error: Missing cloud worker outputs. Run 'make apply' first."; \
		exit 1; \
	fi
	@echo "Logging in to ECR ($(WORKER_REGION))..."
	aws ecr get-login-password --region $(WORKER_REGION) | docker login --username AWS --password-stdin $$(echo $(ECR_REPO) | cut -d/ -f1)
	@echo "Building cloud worker image..."
	docker build -f cloud_worker/Dockerfile -t $(ECR_REPO):latest cloud_worker
	@echo "Pushing cloud worker image..."
	docker push $(ECR_REPO):latest
	@echo "Forcing ECS rollout..."
	aws ecs update-service \
		--region $(WORKER_REGION) \
		--cluster $(ECS_CLUSTER) \
		--service $(ECS_SERVICE) \
		--force-new-deployment > /dev/null
	@echo "$(GREEN)Cloud worker deployed!$(NC)"

deploy-frontend: ## Deploy frontend to S3
	@echo "$(YELLOW)Deploying frontend...$(NC)"
	$(eval BUCKET := $(shell cd $(TERRAFORM_DIR) && terraform output -raw frontend_bucket 2>/dev/null))
	$(eval DIST_ID := $(shell cd $(TERRAFORM_DIR) && terraform output -raw cloudfront_distribution_id 2>/dev/null || echo ""))
	@if [ -z "$(BUCKET)" ]; then \
		echo "Error: Could not get S3 bucket. Run 'make apply' first."; \
		exit 1; \
	fi
	@echo "Building calm-reader frontend..."
	cd calm-reader && npm ci && npm run build
	@echo "Syncing files to S3 ($(BUCKET))..."
	aws s3 sync calm-reader/dist/ s3://$(BUCKET)/ --delete
	# Runtime config used by calm-reader
	aws s3 cp $(FRONTEND_DIR)/js/config.js s3://$(BUCKET)/static/js/
	@if [ -n "$(DIST_ID)" ]; then \
		echo "Invalidating CloudFront cache..."; \
		aws cloudfront create-invalidation --distribution-id $(DIST_ID) --paths "/*" > /dev/null; \
	fi
	@echo "$(GREEN)Frontend deployed!$(NC)"
	@echo "$(GREEN)Visit: https://reader.psybytes.com$(NC)"

# =============================================================================
# Utilities
# =============================================================================

ssh: ## SSH into EC2 instance
	$(eval EC2_IP := $(shell cd $(TERRAFORM_DIR) && terraform output -raw ec2_public_ip 2>/dev/null))
	@if [ -z "$(EC2_IP)" ]; then \
		echo "Error: Could not get EC2 IP. Run 'make apply' first."; \
		exit 1; \
	fi
	ssh -i $(SSH_KEY_PATH) ec2-user@$(EC2_IP)

logs: ## View coordinator logs on EC2
	$(eval EC2_IP := $(shell cd $(TERRAFORM_DIR) && terraform output -raw ec2_public_ip 2>/dev/null))
	ssh -i $(SSH_KEY_PATH) ec2-user@$(EC2_IP) "sudo journalctl -u coordinator -f"

status: ## Check coordinator service status
	$(eval EC2_IP := $(shell cd $(TERRAFORM_DIR) && terraform output -raw ec2_public_ip 2>/dev/null))
	ssh -i $(SSH_KEY_PATH) ec2-user@$(EC2_IP) "sudo systemctl status coordinator"

create-user: ## Create a new Cognito user (usage: make create-user EMAIL=user@example.com)
	@if [ -z "$(EMAIL)" ]; then \
		echo "Usage: make create-user EMAIL=user@example.com"; \
		exit 1; \
	fi
	$(eval POOL_ID := $(shell cd $(TERRAFORM_DIR) && terraform output -raw cognito_user_pool_id 2>/dev/null))
	aws cognito-idp admin-create-user \
		--user-pool-id $(POOL_ID) \
		--username $(EMAIL) \
		--user-attributes Name=email,Value=$(EMAIL) Name=email_verified,Value=true \
		--region $$(cd $(TERRAFORM_DIR) && terraform output -raw aws_region 2>/dev/null || echo "us-east-2")
	@echo "$(GREEN)User created! They will receive an email with temporary password.$(NC)"

clean: ## Clean local build artifacts
	rm -rf $(TERRAFORM_DIR)/.terraform
	rm -rf $(TERRAFORM_DIR)/terraform.tfstate*
	rm -rf __pycache__
	find . -name "*.pyc" -delete

watch-jobs: ## Watch coordinator/worker/SQS + session ETA (per-message, remaining, total) (Ctrl+C to stop)
	python3 scripts/watch_audio_jobs.py

worker-up: ## Run local worker stack (usage: make worker-up WORKER_CONTAINERS=4)
	@if [ "$(WORKER_CONTAINERS)" -lt 1 ]; then \
		echo "WORKER_CONTAINERS must be >= 1"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Starting local worker API + $(WORKER_CONTAINERS) worker container(s)...$(NC)"
	docker compose --profile workers up -d --build --scale epub-reader-worker=$(WORKER_CONTAINERS) epub-reader epub-reader-worker
	@echo "$(GREEN)Workers started. Token sync endpoint: http://127.0.0.1:5001/worker/token$(NC)"

worker-down: ## Stop local worker containers
	docker compose --profile workers stop epub-reader epub-reader-worker

worker-logs: ## Tail API worker logs
	docker compose logs -f epub-reader

worker-status: ## Show local worker containers
	docker compose --profile workers ps
