variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-2"
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

variable "worker_region" {
  description = "Region where ECS GPU cloud worker runs"
  type        = string
  default     = "us-east-2"
}

variable "worker_instance_type" {
  description = "EC2 instance type for ECS worker capacity"
  type        = string
  default     = "g5.xlarge"
}

variable "worker_max_instances" {
  description = "Maximum number of GPU instances for worker capacity"
  type        = number
  default     = 1
}

variable "worker_task_cpu" {
  description = "CPU units for cloud worker task definition"
  type        = number
  default     = 4096
}

variable "worker_task_memory" {
  description = "Task memory (MiB) for cloud worker task definition"
  type        = number
  default     = 15360
}

variable "worker_idle_max_empty_receives" {
  description = "How many empty SQS polls before worker exits"
  type        = number
  default     = 10
}

variable "worker_sqs_wait_seconds" {
  description = "SQS long-poll wait seconds for cloud worker"
  type        = number
  default     = 20
}

variable "worker_m4b_bitrate" {
  description = "AAC bitrate for generated m4b files"
  type        = string
  default     = "64k"
}

variable "worker_qwen_speaker" {
  description = "Qwen speaker preset for TTS synthesis"
  type        = string
  default     = "Ryan"
}

variable "worker_qwen_language" {
  description = "Qwen language selection"
  type        = string
  default     = "Auto"
}

variable "worker_qwen_instruct" {
  description = "Qwen prompt/instruct for stable audiobook voice"
  type        = string
  default     = "Professional audiobook narration. Warm, clear female voice. Natural pacing with brief pauses at commas and sentence ends. Smooth phrasing, no exaggeration. Consistent volume and tone. Crisp consonants, gentle sibilance, minimal breath noise. Slightly slower than conversational speech. Maintain a calm, engaging storyteller tone."
}
