variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "eu-west-1"
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "epub-reader"
}

variable "domain_name" {
  description = "Main domain name"
  type        = string
  default     = "psybytes.com"
}

variable "frontend_subdomain" {
  description = "Frontend subdomain"
  type        = string
  default     = "reader"
}

variable "api_subdomain" {
  description = "API subdomain"
  type        = string
  default     = "api.reader"
}

variable "ec2_instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.small"
}

variable "ssh_public_key_path" {
  description = "Path to SSH public key"
  type        = string
  default     = "~/.ssh/epub_reader.pub"
}

variable "audio_sqs_visibility_timeout" {
  description = "SQS visibility timeout for audio jobs (seconds)"
  type        = number
  default     = 180
}

variable "audio_sqs_receive_wait_seconds" {
  description = "SQS receive wait time (long polling, seconds)"
  type        = number
  default     = 2
}

variable "audio_sqs_max_receive_count" {
  description = "How many receives before message goes to DLQ"
  type        = number
  default     = 8
}
