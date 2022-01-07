import boto3
import pandas as pd
import logger

class amazon_s3:

    def __init__(self,file,bucket_name):
        self.log_path=file
        self.log_writer=logger.App_Logger()
        self.bucket_name=bucket_name

    def create_bucket(self):
        try:
            client = boto3.client(
                service_name='s3',
                region_name='us-east-2',
                aws_access_key_id='AKIAZONLJ4HZHJEHNRYY',
                aws_secret_access_key='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'
            )

            s3 = boto3.resource(
                service_name='s3',
                region_name='us-east-2',
                aws_access_key_id='AKIAZONLJ4HZHJEHNRYY',
                aws_secret_access_key='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'
            )
            buckets=[]
            for bucket in s3.buckets.all():
                buckets.append(bucket.name)


            if self.bucket_name not in buckets:
                response=client.create_bucket(ACL='private',
                                              Bucket=self.bucket_name,
                                              CreateBucketConfiguration={
                                                  'LocationConstraint':'us-east-2'

                                                })
        except Exception as e:
            self.log_writer.log(self.log_path, e)
    def upload(self,file,key):
        try:
            s3 = boto3.resource(
                service_name='s3',
                region_name='us-east-2',
                aws_access_key_id='AKIAZONLJ4HZHJEHNRYY',
                aws_secret_access_key='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'
            )
            s3.Bucket(self.bucket_name).upload_file(Filename=file,Key=key)

        except Exception as e:
            self.log_writer.log(self.log_path, e)
