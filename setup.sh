#!/usr/bin/env bash
# =============================================================================
# MOSS-TTS llama.cpp setup script
# Выполняет все шаги из MOSS-TTS-llama-cpp-guide.md
#
# Использование:
#   bash setup-moss-tts.sh [--install-dir DIR] [--hf-token TOKEN] [--skip-cudnn] [--cpu-only]
#
# Примеры:
#   bash setup-moss-tts.sh --hf-token hf_abc123
#   bash setup-moss-tts.sh --install-dir /opt/moss --hf-token hf_abc123
#   bash setup-moss-tts.sh --hf-token hf_abc123 --skip-cudnn
#   bash setup-moss-tts.sh --hf-token hf_abc123 --cpu-only
# =============================================================================

set -euo pipefail

# -------------------------------- defaults -----------------------------------
INSTALL_DIR="$(pwd)/moss-tts-setup"
HF_TOKEN="${HF_TOKEN:-}"
SKIP_CUDNN=false
CPU_ONLY=false

# -------------------------------- parse args ---------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-dir)  INSTALL_DIR="$2"; shift 2 ;;
        --hf-token)     HF_TOKEN="$2";    shift 2 ;;
        --skip-cudnn)   SKIP_CUDNN=true;  shift ;;
        --cpu-only)     CPU_ONLY=true;     shift ;;
        -h|--help)
            echo "Usage: $0 [--install-dir DIR] [--hf-token TOKEN] [--skip-cudnn] [--cpu-only]"
            echo ""
            echo "  --install-dir DIR    Каталог установки (по умолчанию: ./moss-tts-setup)"
            echo "  --hf-token TOKEN     Hugging Face токен (или задайте HF_TOKEN)"
            echo "  --skip-cudnn         Пропустить установку cuDNN (требует sudo)"
            echo "  --cpu-only           Собрать llama.cpp без CUDA"
            exit 0
            ;;
        *) echo "Неизвестный аргумент: $1"; exit 1 ;;
    esac
done

# -------------------------------- checks -------------------------------------
if [[ -z "$HF_TOKEN" ]]; then
    echo "ОШИБКА: укажите Hugging Face токен через --hf-token или переменную HF_TOKEN"
    echo ""
    echo "  1. Создайте токен: https://huggingface.co/settings/tokens"
    echo "  2. Примите условия доступа:"
    echo "     - https://huggingface.co/OpenMOSS-Team/MOSS-TTS-GGUF"
    echo "     - https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX"
    echo "  3. Запустите: $0 --hf-token hf_ваш_токен"
    exit 1
fi

export HF_TOKEN

# -------------------------------- helpers ------------------------------------
step=0
total_steps=8
log() {
    step=$((step + 1))
    echo ""
    echo "====================================================================="
    echo "  [$step/$total_steps] $1"
    echo "====================================================================="
    echo ""
}

check_command() {
    if ! command -v "$1" &>/dev/null; then
        echo "ОШИБКА: $1 не найден. Установите его и повторите."
        exit 1
    fi
}

check_command git
check_command cmake
check_command gcc
check_command python3

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 10 ]]; then
    echo "ОШИБКА: требуется Python >= 3.10, найден $PYTHON_VERSION"
    exit 1
fi

echo "Каталог установки: $INSTALL_DIR"
echo "Python: $PYTHON_VERSION"
echo "CUDA: $( [[ "$CPU_ONLY" == true ]] && echo 'отключён (--cpu-only)' || echo 'включён' )"

mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# -------------------------------- step 1 -------------------------------------
log "Клонирование MOSS-TTS"

if [[ -d "MOSS-TTS/.git" ]]; then
    echo "MOSS-TTS уже склонирован, пропускаем."
else
    git clone https://github.com/IlyaNizamov/MOSS-TTS-NS.git MOSS-TTS
fi

cd MOSS-TTS
git submodule update --init --recursive
cd "$INSTALL_DIR"

# -------------------------------- step 2 -------------------------------------
log "Создание виртуального окружения"

if [[ -d "MOSS-TTS/venv" ]]; then
    echo "venv уже существует, пропускаем создание."
else
    python3 -m venv MOSS-TTS/venv
fi

MOSS-TTS/venv/bin/pip install --upgrade pip

# -------------------------------- step 3 -------------------------------------
log "Установка Python-зависимостей"

MOSS-TTS/venv/bin/pip install -e "MOSS-TTS[llama-cpp-onnx]"

# -------------------------------- step 4 -------------------------------------
log "Установка cuDNN 9"

if [[ "$SKIP_CUDNN" == true ]] || [[ "$CPU_ONLY" == true ]]; then
    echo "Пропущено ($(  [[ "$CPU_ONLY" == true ]] && echo '--cpu-only' || echo '--skip-cudnn'  ))."
    echo "ONNX-декодер будет работать на CPU."
elif ldconfig -p 2>/dev/null | grep -q "libcudnn.so.9" || \
     ls /usr/lib/x86_64-linux-gnu/libcudnn.so.9* &>/dev/null || \
     ls /usr/local/cuda/lib64/libcudnn.so.9* &>/dev/null; then
    echo "cuDNN 9 уже установлен."
else
    echo "Установка cuDNN 9 (потребуется sudo)..."
    sudo apt-get update -qq
    sudo apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12
    echo "cuDNN 9 установлен."
fi

# -------------------------------- step 5 -------------------------------------
log "Скачивание весов модели"

echo "Скачивание GGUF backbone..."
MOSS-TTS/venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'OpenMOSS-Team/MOSS-TTS-GGUF',
    local_dir='MOSS-TTS/weights/MOSS-TTS-GGUF'
)
print('GGUF — готово')
"

echo ""
echo "Скачивание ONNX аудио-токенизатора..."
MOSS-TTS/venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX',
    local_dir='MOSS-TTS/weights/MOSS-Audio-Tokenizer-ONNX'
)
print('ONNX — готово')
"

# -------------------------------- step 6 -------------------------------------
log "Сборка llama.cpp"

if [[ -d "llama.cpp/.git" ]]; then
    echo "llama.cpp уже склонирован."
else
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
fi

CMAKE_ARGS=(-S llama.cpp -B llama.cpp/build -DCMAKE_BUILD_TYPE=Release)
if [[ "$CPU_ONLY" == false ]]; then
    CMAKE_ARGS+=(-DGGML_CUDA=ON)
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build llama.cpp/build -j"$(nproc)"

echo "llama.cpp собран."

# -------------------------------- step 7 -------------------------------------
log "Сборка C-bridge"

cd MOSS-TTS
bash moss_tts_delay/llama_cpp/build_bridge.sh "$INSTALL_DIR/llama.cpp"
cd "$INSTALL_DIR"

echo "C-bridge собран."

# -------------------------------- step 8 -------------------------------------
log "Проверка — тестовый инференс"

CONFIG="configs/llama_cpp/default.yaml"
if [[ "$CPU_ONLY" == true ]]; then
    CONFIG="configs/llama_cpp/cpu-only.yaml"
fi

cd MOSS-TTS
venv/bin/python -m moss_tts_delay.llama_cpp \
    --config "$CONFIG" \
    --text "Hello! MOSS TTS is working." \
    --output test_output.wav

cd "$INSTALL_DIR"

echo ""
echo "====================================================================="
echo "  УСТАНОВКА ЗАВЕРШЕНА"
echo "====================================================================="
echo ""
echo "Тестовый файл: $INSTALL_DIR/MOSS-TTS/test_output.wav"
echo ""
echo "Для генерации речи:"
echo ""
echo "  cd $INSTALL_DIR/MOSS-TTS"
echo "  source venv/bin/activate"
echo "  python -m moss_tts_delay.llama_cpp \\"
echo "      --config configs/llama_cpp/default.yaml \\"
echo "      --text \"Ваш текст\" \\"
echo "      --output output.wav"
echo ""
