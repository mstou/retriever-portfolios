export CUDA_VISIBLE_DEVICES=0,1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1           
export NCCL_SOCKET_IFNAME=lo       
export VLLM_WORKER_MULTIPROC_METHOD=spawn


python -m vllm.entrypoints.openai.api_server \
  --model "${GEMMA27B_MODEL_DIR:-${MODELS_DIR:-models}/gemma-3-27b-it}" \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 --port 5000
