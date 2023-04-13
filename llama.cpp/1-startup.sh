# #!/usr/bin/env sh
# 1. build via cmake
# 2. get model weights
# 3. build tokens
python3 convert-pth-to-ggml.py models/LLaMA/7B  1
# ./

./build/bin/simple -m models/LLaMA/7B/ggml-model-f16.bin
