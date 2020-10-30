#!/bin/bash

aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_0.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_1.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_2.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_3.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_4.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_5.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_6.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_7.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_8.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_9.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_a.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_b.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_c.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_d.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_e.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/train_f.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/validation.tar.gz .
aws s3 --no-sign-request cp s3://open-images-dataset/tar/test.tar.gz .

tar -xf train_0.tar.gz -C /data/open-images-dataset/train --strip-components 1 --checkpoint=.10000
tar -xf train_1.tar.gz -C /data/open-images-dataset/train --strip-components 1 --checkpoint=.10000
tar -xf train_2.tar.gz -C /data/open-images-dataset/train --strip-components 1 --checkpoint=.10000
tar -xf train_3.tar.gz -C /data/open-images-dataset/train --strip-components 1 --checkpoint=.10000
tar -xf train_4.tar.gz -C /data/open-images-dataset/train --strip-components 1 --checkpoint=.10000
tar -xf train_5.tar.gz -C /data/open-images-dataset/train --strip-components 1 --checkpoint=.10000
tar -xf train_6.tar.gz -C /data/open-images-dataset/train --strip-components 1 --checkpoint=.10000
tar -xf train_7.tar.gz -C /data/open-images-dataset/train --strip-components 1 --checkpoint=.10000
tar -xf train_8.tar.gz -C /data/open-images-dataset/train --strip-components 1 --checkpoint=.10000
tar -xf train_9.tar.gz -C /data/open-images-dataset/train --strip-components 1 --checkpoint=.10000
tar -xf train_a.tar.gz -C /data/open-images-dataset/train --strip-components 1 --checkpoint=.10000
tar -xf train_b.tar.gz -C /data/open-images-dataset/train --strip-components 1 --checkpoint=.10000
tar -xf train_c.tar.gz -C /data/open-images-dataset/train --strip-components 1 --checkpoint=.10000
tar -xf train_d.tar.gz -C /data/open-images-dataset/train --strip-components 1 --checkpoint=.10000
tar -xf train_e.tar.gz -C /data/open-images-dataset/train --strip-components 1 --checkpoint=.10000
tar -xf train_f.tar.gz -C /data/open-images-dataset/train --strip-components 1 --checkpoint=.10000
tar -xf validation.tar.gz -C /data/open-images-dataset/validation --strip-components 1 --checkpoint=.10000
tar -xf test.tar.gz -C /data/open-images-dataset/test --strip-components 1 --checkpoint=.10000