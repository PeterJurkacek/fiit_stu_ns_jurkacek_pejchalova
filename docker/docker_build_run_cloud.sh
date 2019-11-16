#!/bin/sh

image_name=pjurkacek/data-science:version-cloud
docker build -t $image_name .

docker run\
 -u $(id -u):$(id -g)\
 -p 8888:8888\
 -p 6006:6006\
 -v /home/xjurkacekp/fiit_stu_ns_jurkacek_pejchalova:/labs\
 -it $image_name
