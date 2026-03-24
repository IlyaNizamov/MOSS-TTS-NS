# MOSS-TTS-NS

Форк [MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) с HTTP-сервером и скриптом автоматического развёртывания.
Запуск через бэкенд llama.cpp без PyTorch (torch-free inference).

---

## Быстрый старт (скрипт)

Скрипт `setup.sh` выполняет все шаги автоматически: клонирование, venv, зависимости, скачивание весов, сборка llama.cpp и C-bridge, тестовый инференс.

### Подготовка

1. Создайте токен на [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (тип: Read)
2. Примите условия доступа на страницах моделей (кнопка **"Agree and access repository"**):
   - [https://huggingface.co/OpenMOSS-Team/MOSS-TTS-GGUF](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-GGUF)
   - [https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX)

### Запуск

```bash
bash setup.sh --hf-token hf_ваш_токен
```

### Параметры скрипта

| Флаг | Описание |
|------|----------|
| `--install-dir DIR` | Каталог установки (по умолчанию `./moss-tts-setup`) |
| `--hf-token TOKEN` | Токен Hugging Face (или через переменную `HF_TOKEN`) |
| `--skip-cudnn` | Пропустить установку cuDNN (не потребует sudo) |
| `--cpu-only` | Собрать llama.cpp без CUDA |

```bash
# Установка в конкретную директорию
bash setup.sh --install-dir /opt/moss --hf-token hf_abc123

# Без sudo (cuDNN не будет установлен, ONNX-декодер будет работать на CPU)
bash setup.sh --hf-token hf_abc123 --skip-cudnn

# CPU-only сборка (без CUDA)
bash setup.sh --hf-token hf_abc123 --cpu-only
```

Скрипт идемпотентный — если что-то уже склонировано или собрано, этот шаг пропускается. В конце запускается тестовый инференс для проверки.

---

## Требования

- **ОС**: Linux (Ubuntu 22.04 / 24.04)
- **Python**: >= 3.10
- **GPU**: NVIDIA с поддержкой CUDA, минимум 1 карта с 8+ ГБ VRAM (протестировано на RTX 5070 Ti 16 ГБ)
- **CUDA Toolkit**: >= 12.x (протестировано на 12.9)
- **cuDNN**: 9.x (для GPU-ускоренного ONNX-декодирования аудио)
- **GCC / Clang**: для сборки C-bridge
- **CMake**: >= 3.14
- **Git**
- **Аккаунт на Hugging Face**: для скачивания gated-моделей

---

## Ручная установка (пошагово)

### Шаг 1. Клонирование репозитория

```bash
git clone https://github.com/IlyaNizamov/MOSS-TTS-NS.git
cd MOSS-TTS-NS
```

Инициализация submodule (аудио-токенизатор). Без него инференс упадёт с ошибкой `ModuleNotFoundError: No module named 'moss_audio_tokenizer.onnx'`:

```bash
git submodule update --init --recursive
```

### Шаг 2. Создание виртуального окружения

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### Шаг 3. Установка зависимостей

| Профиль | Команда | Для чего |
|---------|---------|----------|
| **ONNX** | `pip install -e ".[llama-cpp-onnx]"` | Рекомендуемый старт, сбалансированная производительность |
| **TensorRT** | `pip install -e ".[llama-cpp-trt]"` | Максимальная скорость аудио-декодера |
| **Torch-accelerated** | `pip install -e ".[llama-cpp-onnx,llama-cpp-torch]"` | GPU-ускоренные LM heads (~30x быстрее) |

```bash
pip install -e ".[llama-cpp-onnx]"
```

### Шаг 4. Установка cuDNN 9

ONNX Runtime использует cuDNN для GPU-ускоренного декодирования аудио. Без cuDNN декодер будет работать на CPU (медленнее, но работает).

```bash
sudo apt-get update
sudo apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12
```

Проверка: `ls /usr/lib/x86_64-linux-gnu/libcudnn.so.9*`

> **Примечание**: первый запуск с GPU-декодером может быть медленным (прогрев ONNX Runtime). Последующие запуски работают быстро.

### Шаг 5. Получение токена Hugging Face

Модели MOSS-TTS на Hugging Face являются **gated** — для скачивания нужна авторизация.

1. Зарегистрируйтесь на [huggingface.co](https://huggingface.co)
2. Создайте токен: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (тип: Read)
3. Примите условия доступа:
   - [https://huggingface.co/OpenMOSS-Team/MOSS-TTS-GGUF](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-GGUF)
   - [https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX)

```bash
export HF_TOKEN=hf_ваш_токен
```

### Шаг 6. Скачивание весов модели

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('OpenMOSS-Team/MOSS-TTS-GGUF', local_dir='weights/MOSS-TTS-GGUF')
snapshot_download('OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX', local_dir='weights/MOSS-Audio-Tokenizer-ONNX')
print('Done')
"
```

> **Примечание**: `huggingface-cli download` может быть недоступен в некоторых версиях `huggingface_hub`. В этом случае используйте Python API как показано выше.

### Шаг 7. Сборка llama.cpp

```bash
cd ..
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j$(nproc)
cd ..
```

> **Для CPU-only**: уберите флаг `-DGGML_CUDA=ON`.

### Шаг 8. Сборка C-bridge

```bash
cd MOSS-TTS-NS
bash moss_tts_delay/llama_cpp/build_bridge.sh ../llama.cpp
```

### Шаг 9. Запуск инференса

```bash
python -m moss_tts_delay.llama_cpp \
    --config configs/llama_cpp/default.yaml \
    --text "Hello, world!" \
    --output output.wav
```

---

## Потребление VRAM и запуск на одной GPU

Квантованная модель (Q4_K_M) занимает ~4.7 ГБ, KV-кэш ~576 МБ, compute buffers ~500 МБ. Итого **~6 ГБ VRAM** — модель свободно помещается на одну карту с 8+ ГБ памяти.

При наличии нескольких GPU llama.cpp автоматически распределит слои между ними. Чтобы ограничить одной картой:

```bash
CUDA_VISIBLE_DEVICES=0 python -m moss_tts_delay.llama_cpp \
    --config configs/llama_cpp/default.yaml \
    --text "Hello, world!" \
    --output output.wav
```

Для карт с 8 ГБ VRAM используйте low-memory конфиг:

```bash
CUDA_VISIBLE_DEVICES=0 python -m moss_tts_delay.llama_cpp \
    --config configs/llama_cpp/trt-8gb.yaml \
    --text "Hello, world!" \
    --output output.wav
```

---

## Конфигурационные файлы

Расположены в `configs/llama_cpp/`:

| Файл | Описание |
|------|----------|
| `default.yaml` | ONNX аудио-бэкенд (рекомендуемый) |
| `trt.yaml` | TensorRT аудио-бэкенд (максимальная скорость) |
| `trt-8gb.yaml` | Staged loading для GPU с ограниченной памятью |
| `cpu-only.yaml` | Полностью CPU |

---

## Выбор модели (квантизация)

В комплекте идут несколько вариантов квантизации backbone-модели. Чем выше квантизация — тем лучше качество, но больше VRAM:

| Модель | Размер | VRAM | Качество |
|--------|--------|------|----------|
| `MOSS_TTS_Q4_K_M.gguf` | 4.8 ГБ | ~6 ГБ | Базовое |
| `MOSS_TTS_Q5_K_M.gguf` | 5.5 ГБ | ~7 ГБ | Лучше |
| `MOSS_TTS_Q6_K.gguf` | 6.3 ГБ | ~8 ГБ | Хорошее |
| `MOSS_TTS_Q8_0.gguf` | 8.2 ГБ | ~10 ГБ | Почти без потерь |
| `MOSS_TTS_F16.gguf` | 16 ГБ | ~18 ГБ | Оригинал (без квантизации) |

По умолчанию используется `Q8_0`. Для смены модели отредактируйте строку `backbone_gguf` в файле `configs/llama_cpp/default.yaml`:

```yaml
# Пример: переключение на Q4
backbone_gguf: weights/MOSS-TTS-GGUF/MOSS_TTS_Q4_K_M.gguf

# Пример: переключение на F16 (оригинал)
backbone_gguf: weights/MOSS-TTS-GGUF/MOSS_TTS_F16.gguf
```

> **Рекомендации**: Q8_0 — лучший баланс качества и скорости для карт с 16 ГБ. Q4_K_M — для карт с 8 ГБ. F16 — если важно максимальное качество и есть 2 GPU.

---

## HTTP-сервер (REST API)

MOSS-TTS можно запустить как HTTP-сервер и генерировать речь через POST-запросы.

### Запуск сервера

```bash
source venv/bin/activate
python server.py --config configs/llama_cpp/default.yaml
```

Сервер стартует на `http://127.0.0.1:8000`. Загрузка модели занимает ~10–15 секунд.

### Параметры сервера

| Флаг | Описание |
|------|----------|
| `--config PATH` | Путь к YAML-конфигу (обязательный) |
| `--host HOST` | Адрес привязки (по умолчанию `127.0.0.1`) |
| `--port PORT` | Порт (по умолчанию `8000`) |
| `--n-gpu-layers N` | Количество слоёв на GPU |
| `--low-memory` | Режим экономии памяти |
| `--reference PATH` | Путь к WAV-файлу с образцом голоса (24 kHz) — фиксирует голос для всех запросов |
| `--seed N` | Random seed по умолчанию — фиксирует голос для всех запросов |

Примеры запуска:

```bash
# Доступ только локально
python server.py --config configs/llama_cpp/default.yaml

# Доступ из сети
python server.py --config configs/llama_cpp/default.yaml --host 0.0.0.0 --port 8080

# На одной GPU
CUDA_VISIBLE_DEVICES=0 python server.py --config configs/llama_cpp/default.yaml

# Фиксированный голос через seed
python server.py --config configs/llama_cpp/default.yaml --seed 42

# Клонирование голоса через reference audio
python server.py --config configs/llama_cpp/default.yaml --reference voice_sample.wav

# Режим экономии памяти (8 ГБ GPU)
CUDA_VISIBLE_DEVICES=0 python server.py --config configs/llama_cpp/trt-8gb.yaml --low-memory
```

### Фиксация голоса

По умолчанию модель генерирует речь **разным голосом** при каждом запросе из-за случайного сэмплирования. Есть два способа зафиксировать голос:

**Способ 1: Seed** — один seed = один голос. Попробуйте разные значения (42, 123, 777...) и выберите понравившийся:

```bash
# На уровне сервера (для всех запросов):
python server.py --config configs/llama_cpp/default.yaml --seed 42

# Или в конкретном запросе (переопределяет серверный):
curl -X POST http://localhost:8000/tts \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello!", "seed": 42}' \
    --output speech.wav
```

> Одинаковый seed + одинаковый текст = побайтово идентичный WAV.

**Способ 2: Reference audio (клонирование голоса)** — передать WAV-образец нужного голоса (24 kHz):

```bash
python server.py --config configs/llama_cpp/default.yaml --reference voice_sample.wav
```

### API эндпоинты

#### `GET /health` — проверка состояния

```bash
curl http://localhost:8000/health
```

Ответ:

```json
{"status": "ok", "model": "MOSS-TTS (llama.cpp)", "sample_rate": 24000}
```

#### `POST /tts` — генерация речи

Принимает JSON, возвращает WAV-файл.

| Поле | Тип | Обязательный | Описание |
|------|-----|:---:|----------|
| `text` | string | да | Текст для синтеза (1–10000 символов) |
| `language` | string | нет | Язык: `zh`, `en` и др. |
| `instruction` | string | нет | Инструкция для генерации |
| `quality` | string | нет | Качество генерации |
| `max_new_tokens` | int | нет | Максимальное число шагов генерации (1–10000) |
| `seed` | int | нет | Random seed для воспроизводимого голоса (переопределяет серверный `--seed`) |
| `format` | string | нет | Формат ответа: `wav` (по умолчанию) или `raw` |

### Примеры запросов

```bash
# Простая генерация
curl -X POST http://localhost:8000/tts \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello, world!"}' \
    --output speech.wav

# С фиксированным голосом (seed)
curl -X POST http://localhost:8000/tts \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello, world!", "seed": 42}' \
    --output speech.wav

# С указанием языка
curl -X POST http://localhost:8000/tts \
    -H "Content-Type: application/json" \
    -d '{"text": "Привет, мир!", "language": "zh"}' \
    --output speech.wav

# С ограничением длины генерации
curl -X POST http://localhost:8000/tts \
    -H "Content-Type: application/json" \
    -d '{"text": "Long text here...", "max_new_tokens": 5000}' \
    --output speech.wav

# Raw PCM (float32, 24kHz, mono)
curl -X POST http://localhost:8000/tts \
    -H "Content-Type: application/json" \
    -d '{"text": "Test", "format": "raw"}' \
    --output speech.pcm
```

**Из Python:**

```python
import requests
import soundfile as sf
import io

response = requests.post(
    "http://localhost:8000/tts",
    json={"text": "Hello from Python!", "seed": 42},
)
response.raise_for_status()

data, sr = sf.read(io.BytesIO(response.content))
print(f"Duration: {len(data)/sr:.2f}s, Sample rate: {sr}")
sf.write("output.wav", data, sr)
```

### Swagger UI

Интерактивная документация: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Возможные проблемы

### `ModuleNotFoundError: No module named 'moss_audio_tokenizer.onnx'`

Не инициализирован git submodule:

```bash
git submodule update --init --recursive
```

### `InvalidRequirement: Parse error` при `pip install`

Устаревший pip:

```bash
pip install --upgrade pip
```

### `GatedRepoError: 401 / 403`

- Убедитесь, что `HF_TOKEN` установлен: `echo $HF_TOKEN`
- Убедитесь, что вы приняли условия на страницах моделей на Hugging Face

### `libcudnn.so.9: cannot open shared object file`

Не установлен cuDNN 9:

```bash
sudo apt-get install -y libcudnn9-cuda-12
```

### Первый запуск с CUDA-декодером очень медленный

Это нормально — ONNX Runtime прогревает GPU. Последующие запуски в рамках одного процесса быстрые.

### Предупреждения о TensorRT

```
EP Error ... Please install TensorRT libraries ...
Falling back to ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

Это не ошибка. ONNX Runtime пытается использовать TensorRT, не находит его и переключается на CUDA.

---

## Структура файлов после установки

```
.
├── llama.cpp/                          # Собранный llama.cpp
│   └── build/bin/libllama.so
├── MOSS-TTS-NS/
│   ├── venv/                           # Виртуальное окружение
│   ├── moss_audio_tokenizer/           # Git submodule
│   ├── moss_tts_delay/
│   │   └── llama_cpp/
│   │       ├── libbackbone_bridge.so   # Собранный C-bridge
│   │       └── ...
│   ├── configs/llama_cpp/
│   │   ├── default.yaml
│   │   ├── trt.yaml
│   │   ├── trt-8gb.yaml
│   │   └── cpu-only.yaml
│   ├── weights/
│   │   ├── MOSS-TTS-GGUF/             # Квантованные веса backbone
│   │   └── MOSS-Audio-Tokenizer-ONNX/ # ONNX encoder/decoder
│   ├── server.py                       # HTTP-сервер (FastAPI)
│   ├── setup.sh                        # Скрипт автоустановки
│   └── output.wav                      # Сгенерированное аудио
```

---

## Ссылки

- Оригинальный репозиторий: [https://github.com/OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS)
- GGUF веса: [https://huggingface.co/OpenMOSS-Team/MOSS-TTS-GGUF](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-GGUF)
- ONNX аудио-токенизатор: [https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX)
- llama.cpp: [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
