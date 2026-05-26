## Data Pipeline
1. load data from clickhouse into a numpy array
```sh
docker exec quant-clickhouse /usr/bin/clickhouse-client \
  --host localhost \
  --port 9000 \
  --user root \
  --password 'pass123' \
  -d quant \
  --send_progress_in_http_headers=0 \
  --query "select [vz5::Float32, vz10::Float32, vz20::Float32, vz30::Float32, vz40::Float32, vz50::Float32, vz60::Float32, vz70::Float32, vz80::Float32, vz90::Float32, vz100::Float32, vz110::Float32, vz120::Float32, vz130::Float32, vz140::Float32, vz150::Float32, vz160::Float32, vz170::Float32, vz180::Float32, vz190::Float32, vz200::Float32] from volume_zscores format Npy" > .clickhouse/share/vzscores32.npy
```
2. shuffle array using `sample-shuffle.py`
3. train / test split shuffled array `train-test-split.py`
4. sub-sample train set `sample-shuffle.py`
5. run pumap on subsampled data ``

