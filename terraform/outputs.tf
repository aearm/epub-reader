output "frontend_url" {
  description = "Frontend URL"
  value       = "https://${local.frontend_domain}"
}

output "api_url" {
  description = "API URL"
  value       = "https://${local.api_domain}"
}

output "ec2_public_ip" {
  description = "EC2 Elastic IP"
  value       = aws_eip.coordinator.public_ip
}

output "ec2_instance_id" {
  description = "EC2 Instance ID"
  value       = aws_instance.coordinator.id
}

output "audio_bucket" {
  description = "S3 bucket for audio files"
  value       = aws_s3_bucket.audio.id
}

output "aws_region" {
  description = "AWS region for deployed resources"
  value       = var.aws_region
}

output "audio_bucket_url" {
  description = "S3 bucket URL for audio files"
  value       = "https://${aws_s3_bucket.audio.bucket_regional_domain_name}"
}

output "audio_sqs_queue_url" {
  description = "Main SQS queue URL for audio generation jobs"
  value       = aws_sqs_queue.audio.url
}

output "audio_sqs_region" {
  description = "Region where audio SQS queue is provisioned"
  value       = var.worker_region
}

output "audio_sqs_dlq_url" {
  description = "DLQ URL for failed audio generation jobs"
  value       = aws_sqs_queue.audio_dlq.url
}

output "audio_sqs_wait_seconds" {
  description = "Coordinator SQS receive wait seconds"
  value       = var.audio_sqs_receive_wait_seconds
}

output "audio_sqs_visibility_timeout" {
  description = "Coordinator SQS visibility timeout seconds"
  value       = var.audio_sqs_visibility_timeout
}

output "audio_sqs_max_receive_count" {
  description = "Coordinator SQS max receive count before DLQ/fail"
  value       = var.audio_sqs_max_receive_count
}

output "frontend_bucket" {
  description = "S3 bucket for frontend"
  value       = aws_s3_bucket.frontend.id
}

output "cognito_user_pool_id" {
  description = "Cognito User Pool ID"
  value       = aws_cognito_user_pool.main.id
}

output "cognito_client_id" {
  description = "Cognito Client ID"
  value       = aws_cognito_user_pool_client.main.id
}

output "cognito_domain" {
  description = "Cognito hosted UI domain"
  value       = "https://${aws_cognito_user_pool_domain.main.domain}.auth.${var.aws_region}.amazoncognito.com"
}

output "ssh_command" {
  description = "SSH command to connect to EC2"
  value       = "ssh -i ~/.ssh/epub_reader ec2-user@${aws_eip.coordinator.public_ip}"
}

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID for cache invalidation"
  value       = aws_cloudfront_distribution.frontend.id
}

output "cloud_worker_ecr_repository_url" {
  description = "ECR repository URL for cloud GPU worker image"
  value       = aws_ecr_repository.cloud_worker.repository_url
}

output "cloud_worker_ecs_cluster_name" {
  description = "ECS cluster name for cloud worker"
  value       = aws_ecs_cluster.cloud_worker.name
}

output "cloud_worker_ecs_service_name" {
  description = "ECS service name for cloud worker"
  value       = aws_ecs_service.cloud_worker.name
}

output "worker_shared_secret" {
  description = "Shared secret used by cloud worker to authenticate to coordinator"
  value       = random_password.worker_shared_secret.result
  sensitive   = true
}
