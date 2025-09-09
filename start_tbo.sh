export SGLANG_TBO_DEBUG=1 
export SGLANG_TBO_FORCE_TBO=1 
export SGLANG_TBO_FORCE_EXTEND=1 
export SGLANG_TBO_SEPARATE_COMM_STREAM=1 
export SGL_CHUNKED_PREFIX_CACHE_THRESHOLD=99999999 

python3 -m sglang.launch_server --model-path /home/yzh/model/Qwen/Qwen2.5-32B-Instruct --trust-remote-code --tp 2 --enable-two-batch-overlap --tbo-delta-extend 1 --tbo-delta-decode 2 --tbo-token-distribution-threshold 0.48 --disable-chunked-prefix-cache --disable-cuda-graph --mem-fraction-static 0.9 --host 0.0.0.0 --port 30000 --disable-radix-cache