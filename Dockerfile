FROM python:3.10

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get install libgomp1


COPY ./web_server/requirement.txt /tmp/requirement.txt
RUN python3 -m pip install -r /tmp/requirement.txt --no-cache-dir


# run server
COPY ./web_server /app
WORKDIR /app
RUN chmod +x ./app/prestart.sh
