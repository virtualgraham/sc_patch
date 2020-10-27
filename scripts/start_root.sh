#!/bin/bash

export AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY
export AWS_DEFAULT_REGION=us-west-2

echo "Starting"

if [ ! -d "/data/home/ubuntu" ] 
then
	echo "No data folder"
	EC2_INSTANCE_ID="`wget -q -O - http://169.254.169.254/latest/meta-data/instance-id || die \"wget instance-id has failed: $?\"`"
	echo $EC2_INSTANCE_ID
	aws ec2 detach-volume --volume-id vol-0fb999bef6150b06a --debug &> /home/ubuntu/foo.out
	sleep 30
	aws ec2 attach-volume --volume-id vol-0fb999bef6150b06a --instance-id $EC2_INSTANCE_ID --device /dev/sdf --debug &> /home/ubuntu/bar.out
	sleep 30  
	mount /dev/nvme2n1p1 /data
fi

echo "runuser"
runuser -l ubuntu -c /data/home/ubuntu/start.sh