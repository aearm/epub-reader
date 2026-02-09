terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Provider for ACM certificates (must be us-east-1 for CloudFront)
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
}

# Provider for GPU worker stack in us-east-2
provider "aws" {
  alias  = "us_east_2"
  region = var.worker_region
}

# Data source for existing Route53 hosted zone
data "aws_route53_zone" "main" {
  name         = var.domain_name
  private_zone = false
}

# Data source for default VPC
data "aws_vpc" "default" {
  default = true
}

# Data source for default subnets
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Random suffix for globally unique S3 bucket names
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

locals {
  frontend_domain = "${var.frontend_subdomain}.${var.domain_name}"
  api_domain      = "${var.api_subdomain}.${var.domain_name}"
  audio_bucket    = "${var.project_name}-audio-${replace(var.aws_region, "-", "")}-${random_id.bucket_suffix.hex}"
  frontend_bucket = "${var.project_name}-frontend-${replace(var.aws_region, "-", "")}-${random_id.bucket_suffix.hex}"
}
