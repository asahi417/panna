# Realtime Img2Img API
This is a realtime image-to-image model hosting API.

## Setup
```shell
pip install -r requirement.txt
```

## Run
```shell
uvicorn app:app --host 0.0.0.0 --port 4444
```
Access API viewer http://0.0.0.0:4444/docs.

## Test
```shell
python test.py
```