# SQS dead-letter queue for failed audio generation messages
resource "aws_sqs_queue" "audio_dlq" {
  provider = aws.us_east_2
  name     = "${var.project_name}-audio-dlq-${random_id.bucket_suffix.hex}"

  # Keep failed messages for up to 14 days for inspection/replay.
  message_retention_seconds = 1209600

  tags = {
    Name    = "${var.project_name}-audio-dlq"
    Project = var.project_name
  }
}

# Main audio generation queue (single queue mode)
resource "aws_sqs_queue" "audio" {
  provider = aws.us_east_2
  name     = "${var.project_name}-audio-jobs-${random_id.bucket_suffix.hex}"

  visibility_timeout_seconds = var.audio_sqs_visibility_timeout
  receive_wait_time_seconds  = var.audio_sqs_receive_wait_seconds

  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.audio_dlq.arn
    maxReceiveCount     = var.audio_sqs_max_receive_count
  })

  tags = {
    Name    = "${var.project_name}-audio-jobs"
    Project = var.project_name
  }
}
