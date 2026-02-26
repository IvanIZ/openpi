SCRIPT_DIR=$(cd -- "$(dirname -- "$BASH_SOURCE[0]")/" && pwd)

cd "$SCRIPT_DIR"
mkdir -p checkpoints/pi05_libero_reason_lora
cd checkpoints/pi05_libero_reason_lora
git clone https://huggingface.co/n5zhong/pi05_libero_reason_lora.git
cd "$SCRIPT_DIR"
mkdir -p assets/pi05_libero_reason_lora
cd assets/pi05_libero_reason_lora
ln -sT ../../checkpoints/pi05_libero_reason_lora/pi05_libero_reason_lora/31799/assets/yilin-wu/ yilin-wu
