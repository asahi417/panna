# Frontend
- setup

```shell
pip install -r requirement.txt
```

- run
```shell
export ENDPOINT="https://tbdavs4mlxmu0k-4000.proxy.runpod.net/,https://tbdavs4mlxmu0k-4444.proxy.runpod.net/"
export P_PROMPT="creative, inspiring, geometric, blooming, surrealistic, HQ"
export N_PROMPT="low quality, blur"
export MAX_CONCURRENT_JOB=6
export HEIGHT=1024
export WIDTH=512
python main.py
```
