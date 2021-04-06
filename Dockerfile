FROM ubuntu
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade && \
    apt-get -y install python3 python3-pip python3-opencv libgtk2.0-dev

COPY *.py /src/
WORKDIR /src

RUN pip3 install flask numpy opencv-python opencv-contrib-python requests
ADD https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names yolo_tiny_configs/coco.names
ADD https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg yolo_tiny_configs/yolov3-tiny.cfg
ADD https://pjreddie.com/media/files/yolov3-tiny.weights yolo_tiny_configs/yolov3-tiny.weights

EXPOSE 5000
CMD ["python3", "/src/iweblens_server.py"]
