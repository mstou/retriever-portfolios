export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1           
export NCCL_SOCKET_IFNAME=lo       
export VLLM_WORKER_MULTIPROC_METHOD=spawn


python -m vllm.entrypoints.openai.api_server \
  --model "${LLAMA70B_MODEL_DIR:-${MODELS_DIR:-models}/llama-3.1-70b-instruct}" \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 --port 8000
