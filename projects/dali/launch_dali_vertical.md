- API
```shell
cd api_server
export HEIGHT=1024
export WIDTH=512
uvicorn app:app --host 0.0.0.0 --port 4444
uvicorn app:app --host 0.0.0.0 --port 4000
```

- Frontend
```shell
cd frontend
export ENDPOINT="https://tbdavs4mlxmu0k-4000.proxy.runpod.net/,https://tbdavs4mlxmu0k-4444.proxy.runpod.net/"
export P_PROMPT="surrealistic, creative, inspiring, geometric, blooming, paint by Salvador Dali, HQ"
export N_PROMPT="low quality, blur, mustache"
export MAX_CONCURRENT_JOB=10
export HEIGHT=1024
export WIDTH=512
python interface_graphic.py
```
