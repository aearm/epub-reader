# -----------------------------------------------------------------------------
# Cloud GPU Worker (ECS EC2 on-demand in us-east-2)
# -----------------------------------------------------------------------------

resource "random_password" "worker_shared_secret" {
  length  = 48
  special = false
}

data "aws_vpc" "worker_default" {
  provider = aws.us_east_2
  default  = true
}

data "aws_subnets" "worker_default" {
  provider = aws.us_east_2
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.worker_default.id]
  }
}

data "aws_ssm_parameter" "ecs_gpu_ami" {
  provider = aws.us_east_2
  name     = "/aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended/image_id"
}

resource "aws_ecr_repository" "cloud_worker" {
  provider = aws.us_east_2
  name     = "${var.project_name}-cloud-worker"

  image_scanning_configuration {
    scan_on_push = true
  }

  force_delete = true

  tags = {
    Name    = "${var.project_name}-cloud-worker-ecr"
    Project = var.project_name
  }
}

resource "aws_cloudwatch_log_group" "cloud_worker" {
  provider          = aws.us_east_2
  name              = "/ecs/${var.project_name}-cloud-worker"
  retention_in_days = 14

  tags = {
    Name    = "${var.project_name}-cloud-worker-logs"
    Project = var.project_name
  }
}

resource "aws_iam_role" "ecs_worker_instance" {
  name = "${var.project_name}-ecs-worker-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name    = "${var.project_name}-ecs-worker-instance-role"
    Project = var.project_name
  }
}

resource "aws_iam_role_policy_attachment" "ecs_worker_instance_ecs" {
  role       = aws_iam_role.ecs_worker_instance.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_role_policy_attachment" "ecs_worker_instance_ssm" {
  role       = aws_iam_role.ecs_worker_instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "ecs_worker_instance" {
  name = "${var.project_name}-ecs-worker-instance-profile"
  role = aws_iam_role.ecs_worker_instance.name
}

resource "aws_security_group" "ecs_worker" {
  provider    = aws.us_east_2
  name        = "${var.project_name}-ecs-worker-sg"
  description = "Security group for ECS GPU worker nodes"
  vpc_id      = data.aws_vpc.worker_default.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.project_name}-ecs-worker-sg"
    Project = var.project_name
  }
}

resource "aws_ecs_cluster" "cloud_worker" {
  provider = aws.us_east_2
  name     = "${var.project_name}-cloud-worker"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name    = "${var.project_name}-cloud-worker-cluster"
    Project = var.project_name
  }
}

resource "aws_launch_template" "ecs_worker_spot" {
  provider      = aws.us_east_2
  name_prefix   = "${var.project_name}-worker-spot-"
  image_id      = data.aws_ssm_parameter.ecs_gpu_ami.value
  instance_type = var.worker_instance_type

  iam_instance_profile {
    name = aws_iam_instance_profile.ecs_worker_instance.name
  }

  vpc_security_group_ids = [aws_security_group.ecs_worker.id]

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = 200
      volume_type           = "gp3"
      delete_on_termination = true
    }
  }

  user_data = base64encode(<<-EOF
    #!/bin/bash
    echo ECS_CLUSTER=${aws_ecs_cluster.cloud_worker.name} >> /etc/ecs/ecs.config
    echo ECS_ENABLE_GPU_SUPPORT=true >> /etc/ecs/ecs.config
    echo ECS_ENABLE_CONTAINER_METADATA=true >> /etc/ecs/ecs.config
  EOF
  )

  update_default_version = true

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name    = "${var.project_name}-cloud-worker-node"
      Project = var.project_name
    }
  }
}

resource "aws_autoscaling_group" "ecs_worker_spot" {
  provider = aws.us_east_2
  name     = "${var.project_name}-worker-spot-${random_id.bucket_suffix.hex}"

  min_size         = 0
  max_size         = var.worker_max_instances
  desired_capacity = 0

  health_check_type   = "EC2"
  vpc_zone_identifier = data.aws_subnets.worker_default.ids

  launch_template {
    id      = aws_launch_template.ecs_worker_spot.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "${var.project_name}-cloud-worker-node"
    propagate_at_launch = true
  }

  tag {
    key                 = "Project"
    value               = var.project_name
    propagate_at_launch = true
  }

  tag {
    key                 = "AmazonECSManaged"
    value               = "true"
    propagate_at_launch = true
  }
}

resource "aws_ecs_capacity_provider" "cloud_worker_spot" {
  provider = aws.us_east_2
  name     = "${var.project_name}-cloud-worker-spot"

  auto_scaling_group_provider {
    auto_scaling_group_arn         = aws_autoscaling_group.ecs_worker_spot.arn
    managed_termination_protection = "DISABLED"
    managed_scaling {
      status                    = "ENABLED"
      target_capacity           = 100
      minimum_scaling_step_size = 1
      maximum_scaling_step_size = 1
      instance_warmup_period    = 180
    }
  }
}

resource "aws_ecs_cluster_capacity_providers" "cloud_worker" {
  provider           = aws.us_east_2
  cluster_name       = aws_ecs_cluster.cloud_worker.name
  capacity_providers = [aws_ecs_capacity_provider.cloud_worker_spot.name]

  default_capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.cloud_worker_spot.name
    weight            = 1
    base              = 0
  }
}

resource "aws_iam_role" "cloud_worker_task_execution" {
  name = "${var.project_name}-cloud-worker-task-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name    = "${var.project_name}-cloud-worker-task-execution-role"
    Project = var.project_name
  }
}

resource "aws_iam_role_policy_attachment" "cloud_worker_task_execution_default" {
  role       = aws_iam_role.cloud_worker_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "cloud_worker_task" {
  name = "${var.project_name}-cloud-worker-task-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name    = "${var.project_name}-cloud-worker-task-role"
    Project = var.project_name
  }
}

resource "aws_iam_role_policy" "cloud_worker_task_sqs" {
  name = "${var.project_name}-cloud-worker-task-sqs"
  role = aws_iam_role.cloud_worker_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:ChangeMessageVisibility",
          "sqs:GetQueueAttributes",
          "sqs:GetQueueUrl"
        ]
        Resource = [
          aws_sqs_queue.audio.arn
        ]
      }
    ]
  })
}

resource "aws_ecs_task_definition" "cloud_worker" {
  provider                 = aws.us_east_2
  family                   = "${var.project_name}-cloud-worker"
  requires_compatibilities = ["EC2"]
  network_mode             = "bridge"
  cpu                      = tostring(var.worker_task_cpu)
  memory                   = tostring(var.worker_task_memory)
  execution_role_arn       = aws_iam_role.cloud_worker_task_execution.arn
  task_role_arn            = aws_iam_role.cloud_worker_task.arn

  container_definitions = jsonencode([
    {
      name      = "cloud-worker"
      image     = "${aws_ecr_repository.cloud_worker.repository_url}:latest"
      essential = true
      cpu       = var.worker_task_cpu
      memory    = var.worker_task_memory
      command   = ["python", "/app/generate_tts.py"]
      resourceRequirements = [
        {
          type  = "GPU"
          value = "1"
        }
      ]
      environment = [
        { name = "COORDINATOR_API_URL", value = "https://${local.api_domain}" },
        { name = "WORKER_SHARED_SECRET", value = random_password.worker_shared_secret.result },
        { name = "AUDIO_SQS_QUEUE_URL", value = aws_sqs_queue.audio.url },
        { name = "AUDIO_SQS_REGION", value = var.worker_region },
        { name = "AUDIO_SQS_WAIT_SECONDS", value = tostring(var.worker_sqs_wait_seconds) },
        { name = "AUDIO_SQS_VISIBILITY_TIMEOUT", value = tostring(var.audio_sqs_visibility_timeout) },
        { name = "WORKER_IDLE_MAX_EMPTY_RECEIVES", value = tostring(var.worker_idle_max_empty_receives) },
        { name = "WORKER_M4B_BITRATE", value = var.worker_m4b_bitrate },
        { name = "QWEN_MODEL_PATH", value = "/models/Qwen3-TTS-12Hz-0.6B-CustomVoice" },
        { name = "QWEN_DEVICE", value = "cuda:0" },
        { name = "QWEN_DTYPE", value = "bfloat16" },
        { name = "QWEN_ATTN_IMPLEMENTATION", value = "flash_attention_2" },
        { name = "QWEN_SPEAKER", value = var.worker_qwen_speaker },
        { name = "QWEN_LANGUAGE", value = var.worker_qwen_language },
        { name = "QWEN_INSTRUCT", value = var.worker_qwen_instruct }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.cloud_worker.name
          awslogs-region        = var.worker_region
          awslogs-stream-prefix = "ecs"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "cloud_worker" {
  provider        = aws.us_east_2
  name            = "${var.project_name}-cloud-worker"
  cluster         = aws_ecs_cluster.cloud_worker.id
  task_definition = aws_ecs_task_definition.cloud_worker.arn
  desired_count   = 0

  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 100
  enable_ecs_managed_tags            = true
  propagate_tags                     = "SERVICE"

  capacity_provider_strategy {
    capacity_provider = aws_ecs_capacity_provider.cloud_worker_spot.name
    weight            = 1
  }

  depends_on = [aws_ecs_cluster_capacity_providers.cloud_worker]
}

resource "aws_appautoscaling_target" "cloud_worker_service" {
  provider           = aws.us_east_2
  max_capacity       = 1
  min_capacity       = 0
  resource_id        = "service/${aws_ecs_cluster.cloud_worker.name}/${aws_ecs_service.cloud_worker.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  service_namespace  = "ecs"
}

resource "aws_appautoscaling_policy" "cloud_worker_scale_out" {
  provider           = aws.us_east_2
  name               = "${var.project_name}-cloud-worker-scale-out"
  policy_type        = "StepScaling"
  resource_id        = aws_appautoscaling_target.cloud_worker_service.resource_id
  scalable_dimension = aws_appautoscaling_target.cloud_worker_service.scalable_dimension
  service_namespace  = aws_appautoscaling_target.cloud_worker_service.service_namespace

  step_scaling_policy_configuration {
    adjustment_type         = "ChangeInCapacity"
    cooldown                = 60
    metric_aggregation_type = "Maximum"

    step_adjustment {
      metric_interval_lower_bound = 0
      scaling_adjustment          = 1
    }
  }
}

resource "aws_appautoscaling_policy" "cloud_worker_scale_in" {
  provider           = aws.us_east_2
  name               = "${var.project_name}-cloud-worker-scale-in"
  policy_type        = "StepScaling"
  resource_id        = aws_appautoscaling_target.cloud_worker_service.resource_id
  scalable_dimension = aws_appautoscaling_target.cloud_worker_service.scalable_dimension
  service_namespace  = aws_appautoscaling_target.cloud_worker_service.service_namespace

  step_scaling_policy_configuration {
    adjustment_type         = "ChangeInCapacity"
    cooldown                = 180
    metric_aggregation_type = "Maximum"

    step_adjustment {
      metric_interval_upper_bound = 0
      scaling_adjustment          = -1
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "cloud_worker_queue_backlog" {
  provider            = aws.us_east_2
  alarm_name          = "${var.project_name}-cloud-worker-queue-backlog"
  alarm_description   = "Scale cloud worker service out when SQS has visible messages"
  comparison_operator = "GreaterThanOrEqualToThreshold"
  evaluation_periods  = 1
  metric_name         = "ApproximateNumberOfMessagesVisible"
  namespace           = "AWS/SQS"
  period              = 60
  statistic           = "Maximum"
  threshold           = 1
  treat_missing_data  = "notBreaching"

  dimensions = {
    QueueName = aws_sqs_queue.audio.name
  }

  alarm_actions = [aws_appautoscaling_policy.cloud_worker_scale_out.arn]
}

resource "aws_cloudwatch_metric_alarm" "cloud_worker_queue_empty" {
  provider            = aws.us_east_2
  alarm_name          = "${var.project_name}-cloud-worker-queue-empty"
  alarm_description   = "Scale cloud worker service in when SQS visible+inflight count reaches zero"
  comparison_operator = "LessThanOrEqualToThreshold"
  evaluation_periods  = 5
  datapoints_to_alarm = 5
  threshold           = 0
  treat_missing_data  = "notBreaching"

  metric_query {
    id = "m1"
    metric {
      metric_name = "ApproximateNumberOfMessagesVisible"
      namespace   = "AWS/SQS"
      period      = 60
      stat        = "Maximum"
      dimensions = {
        QueueName = aws_sqs_queue.audio.name
      }
    }
    return_data = false
  }

  metric_query {
    id = "m2"
    metric {
      metric_name = "ApproximateNumberOfMessagesNotVisible"
      namespace   = "AWS/SQS"
      period      = 60
      stat        = "Maximum"
      dimensions = {
        QueueName = aws_sqs_queue.audio.name
      }
    }
    return_data = false
  }

  metric_query {
    id          = "e1"
    expression  = "m1 + m2"
    label       = "TotalBacklog"
    return_data = true
  }

  alarm_actions = [aws_appautoscaling_policy.cloud_worker_scale_in.arn]
}
