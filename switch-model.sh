#!/usr/bin/env bash
# Переключение квантизации модели в default.yaml
#
# Использование:
#   bash switch-model.sh          # интерактивный выбор
#   bash switch-model.sh q8       # переключить напрямую

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$SCRIPT_DIR/MOSS-TTS/configs/llama_cpp/default.yaml"

# Порядок отображения (от большей к меньшей)
MODELS_ORDER=(f16 q8 q6 q5 q4)

declare -A MODEL_MAP=(
    [f16]="MOSS_TTS_F16.gguf"
    [q4]="MOSS_TTS_Q4_K_M.gguf"
    [q5]="MOSS_TTS_Q5_K_M.gguf"
    [q6]="MOSS_TTS_Q6_K.gguf"
    [q8]="MOSS_TTS_Q8_0.gguf"
)

if [[ ! -f "$CONFIG" ]]; then
    echo "ОШИБКА: конфиг не найден: $CONFIG"
    echo "Сначала запустите setup.sh"
    exit 1
fi

# Текущий GGUF из конфига
CURRENT=$(grep '^backbone_gguf:' "$CONFIG" | head -1)
CURRENT_FILE=$(basename "$CURRENT")
CURRENT_DIR=$(dirname "${CURRENT#backbone_gguf: }")

# Определяем короткое имя текущей модели
current_name="?"
for key in "${!MODEL_MAP[@]}"; do
    if [[ "$CURRENT_FILE" == "${MODEL_MAP[$key]}" ]]; then
        current_name="$key"
        break
    fi
done

apply_model() {
    local model="$1"
    local gguf_file="${MODEL_MAP[$model]}"
    local new_path="$CURRENT_DIR/$gguf_file"

    if [[ ! -f "$new_path" ]]; then
        echo "ОШИБКА: файл не найден: $new_path"
        exit 1
    fi

    sed -i "s|^backbone_gguf:.*|backbone_gguf: ${new_path}|" "$CONFIG"
    echo "Переключено: $current_name -> $model ($gguf_file)"
}

# Прямой вызов с аргументом
if [[ $# -ge 1 ]]; then
    MODEL="$1"
    if [[ -z "${MODEL_MAP[$MODEL]:-}" ]]; then
        echo "ОШИБКА: неизвестная модель '$MODEL'"
        echo "Допустимые: ${MODELS_ORDER[*]}"
        exit 1
    fi
    if [[ "$MODEL" == "$current_name" ]]; then
        echo "Модель $MODEL уже выбрана."
        exit 0
    fi
    apply_model "$MODEL"
    exit 0
fi

# Интерактивный режим
echo "Текущая модель: $current_name ($CURRENT_FILE)"
echo ""
echo "Доступные модели:"

for i in "${!MODELS_ORDER[@]}"; do
    key="${MODELS_ORDER[$i]}"
    file="${MODEL_MAP[$key]}"
    num=$((i + 1))
    full_path="$CURRENT_DIR/$file"

    marker="  "
    [[ "$key" == "$current_name" ]] && marker="> "

    if [[ -f "$full_path" ]]; then
        size=$(du -hL "$full_path" | cut -f1)
        printf "  %s%d) %-4s  %-24s  %s\n" "$marker" "$num" "$key" "$file" "$size"
    else
        printf "  %s%d) %-4s  %-24s  %s\n" "$marker" "$num" "$key" "$file" "(не скачан)"
    fi
done

echo ""
read -rp "Выберите модель [1-${#MODELS_ORDER[@]}] (Enter — отмена): " choice

if [[ -z "$choice" ]]; then
    echo "Отменено."
    exit 0
fi

if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#MODELS_ORDER[@]} )); then
    echo "ОШИБКА: введите число от 1 до ${#MODELS_ORDER[@]}"
    exit 1
fi

selected="${MODELS_ORDER[$((choice - 1))]}"

if [[ "$selected" == "$current_name" ]]; then
    echo "Модель $selected уже выбрана."
    exit 0
fi

apply_model "$selected"
