docker run\
 -u $(id -u):$(id -g)\
 --rm\
 -p 8888:8888\
 -p 6006:6006\
 -v /Users/peterjurkacek/STU_FIIT_OFFLINE/ns/fiit_stu_ns_jurkacek_pejchalova:/labs\
 --name datascience\
 -it $image_name