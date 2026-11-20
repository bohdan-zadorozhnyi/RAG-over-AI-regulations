terraform {
  required_version = ">= 1.9"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.60" }
  }
}

provider "aws" {
  region = var.region
}

# No default on purpose: Terraform refuses to run without an explicit region.
# Set it in terraform.tfvars (see terraform.tfvars.example).
variable "region" {
  type = string
}

variable "project" {
  type    = string
  default = "reg-rag"
}

# Raw corpus. Versioned because regulation gets amended and an answer must be
# traceable to the text it was generated from.
resource "aws_s3_bucket" "raw" {
  bucket_prefix = "${var.project}-raw-"
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "raw" {
  bucket = aws_s3_bucket.raw.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_public_access_block" "raw" {
  bucket                  = aws_s3_bucket.raw.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Aurora Serverless v2 Postgres. On the AWS free tier via express
# configuration: up to 4 ACUs and 1 GB per cluster, max 2 clusters.
# TODO step 8: aws_rds_cluster + aws_rds_cluster_instance, pgvector enabled,
# credentials in Secrets Manager, min_capacity 0.5 so it scales down.

# TODO step 8: IAM role granting bedrock:InvokeModel for the embedding model
# only, and s3:GetObject on this bucket. No wildcards.

output "raw_bucket" {
  value = aws_s3_bucket.raw.bucket
}
