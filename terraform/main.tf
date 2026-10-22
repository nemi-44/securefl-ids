terraform {
  required_version = ">= 1.5"
  required_providers {
    aws    = { source = "hashicorp/aws", version = "5.72.1" }
    random = { source = "hashicorp/random", version = "3.6.3" }
  }
}

provider "aws" {
  region = var.region
  default_tags { tags = { Project = "securefl-ids" } }
}

variable "region" { default = "eu-west-1" }
variable "instance_type" { default = "t3.small" }

resource "random_id" "sfx" { byte_length = 4 }

# Parameter bus: global models, client updates, shards and results
resource "aws_s3_bucket" "bus" {
  bucket        = "securefl-ids-bus-${random_id.sfx.hex}"
  force_destroy = true
}
resource "aws_s3_bucket_public_access_block" "bus" {
  bucket                  = aws_s3_bucket.bus.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
resource "aws_s3_bucket_server_side_encryption_configuration" "bus" {
  bucket = aws_s3_bucket.bus.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "AES256" }
  }
}

resource "aws_cloudwatch_log_group" "fl" {
  name              = "/securefl-ids/runs"
  retention_in_days = 14
}

data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}
resource "aws_iam_role" "node" {
  name               = "securefl-ids-node-${random_id.sfx.hex}"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}
resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}
resource "aws_iam_role_policy" "node" {
  role = aws_iam_role.node.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket", "s3:DeleteObject"],
      Resource = [aws_s3_bucket.bus.arn, "${aws_s3_bucket.bus.arn}/*"] },
      { Effect = "Allow", Action = ["cloudwatch:PutMetricData"], Resource = "*" },
      { Effect = "Allow", Action = ["logs:CreateLogStream", "logs:PutLogEvents"], Resource = "${aws_cloudwatch_log_group.fl.arn}:*" }
    ]
  })
}
resource "aws_iam_instance_profile" "node" {
  name = "securefl-ids-node-${random_id.sfx.hex}"
  role = aws_iam_role.node.name
}

data "aws_ami" "al2023" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["al2023-ami-2023*-x86_64"]
  }
}
data "aws_vpc" "default" { default = true }

resource "aws_security_group" "node" {
  name        = "securefl-ids-node-${random_id.sfx.hex}"
  description = "No inbound; outbound HTTPS for S3, CloudWatch and SSM"
  vpc_id      = data.aws_vpc.default.id
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "node" {
  ami                    = data.aws_ami.al2023.id
  instance_type          = var.instance_type
  iam_instance_profile   = aws_iam_instance_profile.node.name
  vpc_security_group_ids = [aws_security_group.node.id]
  metadata_options { http_tokens = "required" }
  root_block_device {
    volume_size = 20
    volume_type = "gp3"
    encrypted   = true
  }
  user_data = <<-EOT
    #!/bin/bash
    dnf install -y docker
    systemctl enable --now docker
  EOT
  tags      = { Name = "securefl-ids-node" }
}

resource "aws_cloudwatch_dashboard" "fl" {
  dashboard_name = "SecureFL-IDS"
  dashboard_body = jsonencode({ widgets = [
    { type = "metric", x = 0, y = 0, width = 12, height = 6, properties = {
      title = "Round time (s)", region = var.region, stat = "Average", period = 60,
    metrics = [["SecureFL-IDS", "RoundSeconds", "Arm", "securefl"], ["...", "fedavg"], ["...", "fedavg_topk"]] } },
    { type = "metric", x = 12, y = 0, width = 12, height = 6, properties = {
      title = "Upload bytes per round", region = var.region, stat = "Sum", period = 60,
    metrics = [["SecureFL-IDS", "UploadBytes", "Arm", "securefl"], ["...", "fedavg"], ["...", "fedavg_topk"]] } },
    { type = "metric", x = 0, y = 6, width = 12, height = 6, properties = {
      title = "EC2 CPU (%)", region = var.region, stat = "Average", period = 60,
    metrics = [["AWS/EC2", "CPUUtilization", "InstanceId", aws_instance.node.id]] } }
  ] })
}

output "bucket" { value = aws_s3_bucket.bus.bucket }
output "instance_id" { value = aws_instance.node.id }
output "log_group" { value = aws_cloudwatch_log_group.fl.name }
