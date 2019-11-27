#!/bin/sh

image_name=pjurkacek/data-science:version1.0.0
docker build -t $image_name .

docker run\
 -u $(id -u):$(id -g)\
 --rm\
 -p 8888:8888\
 -p 6006:6006\
 -v /home/xjurkacekp/fiit_stu_ns_jurkacek_pejchalova:/labs\
 --name datascience\
 -it $image_name