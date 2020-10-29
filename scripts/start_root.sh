#!/bin/bash

export AWS_ACCESS_KEY_ID=00000000000000000
export AWS_SECRET_ACCESS_KEY=00000000000000000
export AWS_DEFAULT_REGION=us-west-2

echo "Starting"

if [[ $(lsblk | grep nvme2n1) ]]; then
	echo "Data drive is attached"
else
	echo "Data drive is not attached"
	EC2_INSTANCE_ID="`wget -q -O - http://169.254.169.254/latest/meta-data/instance-id || die \"wget instance-id has failed: $?\"`"
	echo $EC2_INSTANCE_ID
	aws ec2 detach-volume --volume-id vol-00000000000000000 --debug &> /home/ubuntu/detach.out
	sleep 30
	aws ec2 attach-volume --volume-id vol-00000000000000000 --instance-id $EC2_INSTANCE_ID --device /dev/sdf --debug &> /home/ubuntu/attach.out
	sleep 30  
fi

if [ -d "/data/sc_patch" ]; then
	echo "Data drive is mounted"
else
	echo "Data drive is not mounted"
	mount /dev/nvme2n1 /data
fi

echo "Starting trainging script"
runuser -l ubuntu -c /data/start.sh