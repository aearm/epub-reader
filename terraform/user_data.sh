#!/bin/bash
set -e

# Update system
yum update -y

# Install Python and dependencies
yum install -y python3.11 python3.11-pip git nginx

# Create app directory with ec2-user ownership
mkdir -p /opt/epub-reader
chown -R ec2-user:ec2-user /opt/epub-reader
cd /opt/epub-reader

# Create environment file
cat > /opt/epub-reader/.env << EOF
AUDIO_BUCKET=${audio_bucket}
AWS_REGION=${aws_region}
COGNITO_POOL_ID=${cognito_pool_id}
COGNITO_CLIENT_ID=${cognito_client_id}
AUDIO_SQS_QUEUE_URL=${audio_sqs_queue_url}
AUDIO_SQS_DLQ_URL=${audio_sqs_dlq_url}
AUDIO_SQS_WAIT_SECONDS=${audio_sqs_wait_seconds}
AUDIO_SQS_VISIBILITY_TIMEOUT=${audio_sqs_visibility_timeout}
AUDIO_SQS_MAX_RECEIVE_COUNT=${audio_sqs_max_receive_count}
EOF

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install Python packages
pip install flask flask-cors boto3 python-jose requests gunicorn

# Create coordinator service
cat > /etc/systemd/system/coordinator.service << EOF
[Unit]
Description=EPUB Reader Coordinator
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/epub-reader
EnvironmentFile=/opt/epub-reader/.env
ExecStart=/opt/epub-reader/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 coordinator:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Configure Nginx as reverse proxy
cat > /etc/nginx/conf.d/coordinator.conf << EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Remove default nginx config
rm -f /etc/nginx/conf.d/default.conf

# Enable and start nginx
systemctl enable nginx
systemctl start nginx

# Note: Coordinator code will be deployed via Makefile
echo "EC2 bootstrap complete. Deploy coordinator code with: make deploy-backend"
