import boto3

s3 = boto3.client("s3")
bucket_name = "mutcd-all-signs-us"

response = s3.list_objects_v2(Bucket=bucket_name, Prefix="")

for obj in response.get("Contents", []):
    if obj["Key"].endswith(".png"):
        print(f"https://{bucket_name}.s3.us-east-1.amazonaws.com/{obj['Key']}")
