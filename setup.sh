#!/usr/bin/env bash
# =============================================================================
# MOSS-TTS llama.cpp setup script
# Выполняет все шаги из MOSS-TTS-llama-cpp-guide.md
#
# Использование:
#   bash setup-moss-tts.sh [--install-dir DIR] [--hf-token TOKEN] [--model MODEL] [--skip-cudnn] [--cpu-only]
#
# Примеры:
#   bash setup-moss-tts.sh --hf-token hf_abc123
#   bash setup-moss-tts.sh --hf-token hf_abc123 --model q8
#   bash setup-moss-tts.sh --install-dir /opt/moss --hf-token hf_abc123
#   bash setup-moss-tts.sh --hf-token hf_abc123 --skip-cudnn
#   bash setup-moss-tts.sh --hf-token hf_abc123 --cpu-only
# =============================================================================

set -euo pipefail

# -------------------------------- defaults -----------------------------------
INSTALL_DIR="$(pwd)"
HF_TOKEN="${HF_TOKEN:-}"
MODEL="q4"
SKIP_CUDNN=false
CPU_ONLY=false

# Маппинг коротких имён моделей на имена GGUF-файлов
declare -A MODEL_MAP=(
    [f16]="MOSS_TTS_F16.gguf"
    [q4]="MOSS_TTS_Q4_K_M.gguf"
    [q5]="MOSS_TTS_Q5_K_M.gguf"
    [q6]="MOSS_TTS_Q6_K.gguf"
    [q8]="MOSS_TTS_Q8_0.gguf"
)

# -------------------------------- parse args ---------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-dir)  INSTALL_DIR="$2"; shift 2 ;;
        --hf-token)     HF_TOKEN="$2";    shift 2 ;;
        --model)        MODEL="$2";       shift 2 ;;
        --skip-cudnn)   SKIP_CUDNN=true;  shift ;;
        --cpu-only)     CPU_ONLY=true;     shift ;;
        -h|--help)
            echo "Usage: $0 [--install-dir DIR] [--hf-token TOKEN] [--model MODEL] [--skip-cudnn] [--cpu-only]"
            echo ""
            echo "  --install-dir DIR    Каталог установки (по умолчанию: текущая директория)"
            echo "  --hf-token TOKEN     Hugging Face токен (или задайте HF_TOKEN)"
            echo "  --model MODEL        Квантизация модели: f16, q4, q5, q6, q8 (по умолчанию: q4)"
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

# Валидация --model
GGUF_FILE="${MODEL_MAP[$MODEL]:-}"
if [[ -z "$GGUF_FILE" ]]; then
    echo "ОШИБКА: неизвестная модель '$MODEL'. Допустимые значения: f16, q4, q5, q6, q8"
    exit 1
fi

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

# -------------------------------- prerequisite checks -------------------------
MISSING=()

# git
if ! command -v git &>/dev/null; then
    MISSING+=("git")
    echo "ОШИБКА: git не найден."
fi

# cmake / g++
if ! command -v cmake &>/dev/null || ! command -v g++ &>/dev/null; then
    echo "ОШИБКА: не найдены cmake и/или g++ — нужны для сборки llama.cpp."
    MISSING+=("cmake/g++")
fi

# python3
if ! command -v python3 &>/dev/null; then
    echo "ОШИБКА: python3 не найден."
    MISSING+=("python3")
else
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
    PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
    if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 10 ]]; then
        echo "ОШИБКА: требуется Python >= 3.10, найден $PYTHON_VERSION"
        MISSING+=("python>=3.10")
    fi

    # venv module
    if ! python3 -m venv --help &>/dev/null 2>&1; then
        echo "ОШИБКА: модуль venv для Python ${PYTHON_VERSION} недоступен."
        MISSING+=("python-venv")
    fi
fi

# CUDA (nvcc) — только если не --cpu-only
if [[ "$CPU_ONLY" == false ]]; then
    if ! command -v nvcc &>/dev/null; then
        echo "ОШИБКА: nvcc не найден — CUDA toolkit не установлен."
        MISSING+=("nvcc")
    fi
fi

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo ""
    echo "Установите недостающие зависимости и повторите запуск."
    exit 1
fi

echo "Каталог установки: $INSTALL_DIR"
echo "Модель: $MODEL ($GGUF_FILE)"
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
log "Проверка cuDNN 9"

if [[ "$SKIP_CUDNN" == true ]] || [[ "$CPU_ONLY" == true ]]; then
    echo "Пропущено ($(  [[ "$CPU_ONLY" == true ]] && echo '--cpu-only' || echo '--skip-cudnn'  ))."
    echo "ONNX-декодер будет работать на CPU."
elif ldconfig -p 2>/dev/null | grep -q "libcudnn.so.9" || \
     ls /usr/lib/x86_64-linux-gnu/libcudnn.so.9* &>/dev/null || \
     ls /usr/local/cuda/lib64/libcudnn.so.9* &>/dev/null; then
    echo "cuDNN 9 уже установлен."
else
    echo "ОШИБКА: cuDNN 9 не найден — нужен для ONNX-декодера на GPU."
    echo "  Перезапустите с --skip-cudnn если GPU-ускорение не требуется."
    exit 1
fi

# -------------------------------- step 5 -------------------------------------
log "Скачивание весов модели"

echo "Скачивание GGUF backbone..."
GGUF_DIR=$(MOSS-TTS/venv/bin/python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('OpenMOSS-Team/MOSS-TTS-GGUF')
print(p)
")
echo "GGUF — готово: $GGUF_DIR"

echo ""
echo "Скачивание ONNX аудио-токенизатора..."
# ONNX модели используют external data (.data файлы) — кеш HF с симлинками
# ломает резолвинг путей в ONNX Runtime, поэтому качаем в local_dir.
ONNX_DIR="MOSS-TTS/weights/MOSS-Audio-Tokenizer-ONNX"
MOSS-TTS/venv/bin/python -c "
from huggingface_hub import snapshot_download
from pathlib import Path
import sys

local_dir = sys.argv[1]
expected = ['encoder.onnx', 'encoder.data', 'decoder.onnx', 'decoder.data']
all_ok = all((Path(local_dir) / f).exists() for f in expected)

if all_ok:
    # Проверяем целостность: сравниваем размеры с метаданными из HF
    from huggingface_hub import repo_info
    info = repo_info('OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX')
    remote_sizes = {s.rfilename: s.size for s in info.siblings}
    for f in expected:
        local_size = (Path(local_dir) / f).stat().st_size
        remote_size = remote_sizes.get(f)
        if remote_size and local_size != remote_size:
            print(f'  {f}: {local_size} != {remote_size} (ожидается), перекачиваем...')
            all_ok = False
            break

if all_ok:
    print('ONNX токенизатор уже скачан, пропускаем.')
else:
    snapshot_download(
        'OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX',
        local_dir=local_dir,
    )
    print('ONNX — готово')
" "$ONNX_DIR"

echo ""
echo "Обновляем пути в конфигах..."
MOSS-TTS/venv/bin/python -c "
import sys, yaml, pathlib

gguf_dir = sys.argv[1]
gguf_file = sys.argv[2]

for cfg_path in pathlib.Path('MOSS-TTS/configs/llama_cpp').glob('*.yaml'):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    changed = False
    mapping = {
        'backbone_gguf':      lambda v: gguf_dir + '/' + pathlib.Path(v).name,
        'embedding_dir':      lambda v: gguf_dir + '/embeddings',
        'lm_head_dir':        lambda v: gguf_dir + '/lm_heads',
        'tokenizer_dir':      lambda v: gguf_dir + '/tokenizer',
    }
    for key, fn in mapping.items():
        if key in cfg:
            new_val = fn(cfg[key])
            if cfg[key] != new_val:
                cfg[key] = new_val
                changed = True
    # Для default.yaml подставляем выбранную модель
    if cfg_path.name == 'default.yaml' and 'backbone_gguf' in cfg:
        desired = gguf_dir + '/' + gguf_file
        if cfg['backbone_gguf'] != desired:
            cfg['backbone_gguf'] = desired
            changed = True
    if changed:
        with open(cfg_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f'  Обновлён: {cfg_path.name}')
" "$GGUF_DIR" "$GGUF_FILE"

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
    # Выбираем правильный nvcc: предпочитаем /usr/local/cuda/bin/nvcc
    if [[ -x /usr/local/cuda/bin/nvcc ]]; then
        echo "Используем CUDA compiler: /usr/local/cuda/bin/nvcc"
        CMAKE_ARGS+=(-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc)
    fi
    # Определяем compute capability GPU автоматически
    CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    if [[ -n "$CUDA_ARCH" ]]; then
        echo "Обнаружена GPU архитектура: sm_${CUDA_ARCH}"
        CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
    fi
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
