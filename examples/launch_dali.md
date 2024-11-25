- API
```shell
cd api_server
export HEIGHT=512
export WIDTH=512
uvicorn app:app --host 0.0.0.0 --port 4444
uvicorn app:app --host 0.0.0.0 --port 4000
```

- Frontend (GPU)
```shell
cd frontend
export ENDPOINT="https://tbdavs4mlxmu0k-4000.proxy.runpod.net/,https://tbdavs4mlxmu0k-4444.proxy.runpod.net/"
export P_PROMPT="surrealistic, creative, inspiring, geometric, blooming, paint by Salvador Dali, HQ"
export N_PROMPT="low quality, blur, mustache"
export MAX_CONCURRENT_JOB=8
export HEIGHT=512
export WIDTH=512
export FPS=120
export WAIT_M_SEC=160
python interface_graphic.py
```


- Frontend (MPS)
```shell
cd frontend
export ENDPOINT="http://0.0.0.0:4444"
export P_PROMPT="surrealistic, creative, inspiring, geometric, blooming, paint by Salvador Dali, HQ"
export N_PROMPT="low quality, blur, mustache"
export MAX_CONCURRENT_JOB=4
export HEIGHT=512
export WIDTH=512
export FPS=120
export WAIT_M_SEC=160
python interface_graphic.py
```
