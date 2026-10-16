cd ~/repos/research/ai-torture-chamber
K=~/.runpod/ssh/runpodctl-ssh-key; H=root@60.249.37.148; P=26204

# 1. push trainer + dataset
scp -i $K -P $P scripts/train_persona_lora.py \
      data/voices/_delusion_blend/delusion_sft.jsonl $H:/root/

# 2. install deps + launch training (runs detached on the pod)
ssh -i $K -p $P $H 'pip install -U -q transformers peft bitsandbytes accelerate && cd /root && \
  PERSONA_BASE=OBLITERATUS/Qwen3.8-27B-OBLITERATED \
  PERSONA_DATA=/root/delusion_sft.jsonl PERSONA_OUT=/root/delusion_adapter \
  nohup python -u train_persona_lora.py > /root/train.log 2>&1 & echo launched'

# 3. watch it (Ctrl-C to stop watching; training keeps running)
ssh -i $K -p $P $H 'tail -f /root/train.log'

When the log prints saved adapter -> /root/delusion_adapter, pull it back and shut the pod down:
ssh -i $K -p $P $H 'cd /root && tar czf delusion_adapter.tgz delusion_adapter'
scp -i $K -P $P $H:/root/delusion_adapter.tgz data/voices/_delusion_blend/
runpodctl pod delete lkglya6akddrc7     # stop billing ($0.79/hr)