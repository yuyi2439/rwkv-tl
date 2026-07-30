uv run python script/compare_rwkv7_speed.py \
  --project-checkpoint ~/rwkv/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --vocab asset/rwkv_vocab_v20230424.txt \
  --fast-script ~/rwkv/Albatross/faster3a_2607 \
  --device cuda \
  --cases 1x1,8x8,16x16