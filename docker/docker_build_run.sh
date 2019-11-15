#!/bin/sh

image_name=pjurkacek/data-science:version0.0

docker build -t $image_name .

docker run\
 -u $(id -u):$(id -g)\
 -p 8888:8888\
 -p 6006:6006\
 -v ~/Google\ Drive\ File\ Stream/Môj\ disk/STU_FIIT/inzinier/3_semester/ns/projekt/fiit_stu_ns_jurkacek_pejchalova:/labs\
 -it $image_name