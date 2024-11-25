- API
```shell
cd api_server
export HEIGHT=512
export WIDTH=512
uvicorn app:app --host 0.0.0.0 --port 4444
uvicorn app:app --host 0.0.0.0 --port 4000
```

- Frontend (Config)
```shell
cd frontend
export ENDPOINT="https://tbdavs4mlxmu0k-4000.proxy.runpod.net/,https://tbdavs4mlxmu0k-4444.proxy.runpod.net/"
python interface_config.py
```
- 
- Frontend (Graphic)
```shell
cd frontend
export ENDPOINT="https://tbdavs4mlxmu0k-4000.proxy.runpod.net/,https://tbdavs4mlxmu0k-4444.proxy.runpod.net/"
export MAX_CONCURRENT_JOB=12
export HEIGHT=512
export WIDTH=512
export FPS=120
export WAIT_M_SEC=120
python interface_graphic.py
```