#!/bin/sh

image_name=pjurkacek/data-science:version1.0.1
docker build -t $image_name .

docker run\
 -u $(id -u):$(id -g)\
 --rm\
 -p 8888:8888\
 -p 6006:6006\
 -v /Users/peterjurkacek/STU_FIIT_OFFLINE/ns/fiit_stu_ns_jurkacek_pejchalova:/labs\
 --name tensorboard_datascience\
 -it $image_name

