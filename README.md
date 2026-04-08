# MOSS-TTS-NS

Форк [MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS) с HTTP-сервером и скриптом автоматического развёртывания.
Запуск через бэкенд llama.cpp без PyTorch (torch-free inference).

---

## Подготовка системы

Перед установкой убедитесь, что в системе присутствуют все необходимые компоненты.

### ОС

Ubuntu 22.04 / 24.04 (или совместимый Linux-дистрибутив).

### Python >= 3.10

```bash
python3 --version
```

Должен быть доступен модуль `venv`. На Ubuntu он ставится отдельно:

```bash
sudo apt install python3-venv
```

### Git

```bash
sudo apt install git
```

### Компиляторы и CMake

Нужны для сборки llama.cpp и C-bridge:

```bash
sudo apt install cmake g++ build-essential
```

### NVIDIA драйвер и CUDA Toolkit

Для GPU-инференса требуется установленный драйвер NVIDIA и CUDA Toolkit >= 12.x.

Проверка драйвера:

```bash
nvidia-smi
```

Проверка CUDA Toolkit (компилятор `nvcc`):

```bash
nvcc --version
```

Если `nvcc` не найден — установите CUDA Toolkit: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

> При сборке с `--cpu-only` драйвер NVIDIA и CUDA Toolkit не требуются.

### cuDNN 9

Нужен для GPU-ускоренного ONNX-декодирования аудио. Без cuDNN декодер будет работать на CPU (медленнее, но работает).

Установите по [инструкции](https://developer.nvidia.com/cudnn-downloads)

Проверка:

```bash
ls /usr/lib/x86_64-linux-gnu/libcudnn.so.9*
```

> Можно пропустить установку cuDNN, запустив `setup.sh` с флагом `--skip-cudnn`.

### Аккаунт Hugging Face

Модели MOSS-TTS на Hugging Face являются **gated** — для скачивания нужна авторизация.

1. Зарегистрируйтесь на [huggingface.co](https://huggingface.co)
2. Создайте токен: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (тип: Read)
3. Примите условия доступа (кнопка **"Agree and access repository"**):
   - [https://huggingface.co/OpenMOSS-Team/MOSS-TTS-GGUF](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-GGUF)
   - [https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX)

---

## Быстрый старт (скрипт)

Скрипт `setup.sh` выполняет все шаги автоматически: клонирование, venv, зависимости, скачивание весов, сборка llama.cpp и C-bridge, тестовый инференс.

```bash
bash setup.sh --hf-token hf_ваш_токен
```

### Параметры скрипта

| Флаг | Описание |
|------|----------|
| `--install-dir DIR` | Каталог установки (по умолчанию: текущая директория) |
| `--hf-token TOKEN` | Токен Hugging Face (или через переменную `HF_TOKEN`) |
| `--model MODEL` | Квантизация модели: `f16`, `q4`, `q5`, `q6`, `q8` (по умолчанию: `q4`) |
| `--skip-cudnn` | Пропустить проверку cuDNN (ONNX-декодер будет на CPU) |
| `--cpu-only` | Собрать llama.cpp без CUDA |

```bash
# Установка с моделью Q8
bash setup.sh --hf-token hf_abc123 --model q8

# Установка в конкретную директорию
bash setup.sh --install-dir /opt/moss --hf-token hf_abc123

# Без cuDNN (декодер на CPU)
bash setup.sh --hf-token hf_abc123 --skip-cudnn

# CPU-only сборка (без CUDA)
bash setup.sh --hf-token hf_abc123 --cpu-only
```

Скрипт идемпотентный — если что-то уже склонировано или собрано, этот шаг пропускается. В конце запускается тестовый инференс для проверки.

---

## Выбор модели (квантизация)

В комплекте идут несколько вариантов квантизации backbone-модели:

| Модель | Файл | Размер | VRAM | Качество |
|--------|------|--------|------|----------|
| `f16` | `MOSS_TTS_F16.gguf` | 16 ГБ | ~18 ГБ | Оригинал (без квантизации) |
| `q8` | `MOSS_TTS_Q8_0.gguf` | 8.2 ГБ | ~10 ГБ | Почти без потерь |
| `q6` | `MOSS_TTS_Q6_K.gguf` | 6.3 ГБ | ~8 ГБ | Хорошее |
| `q5` | `MOSS_TTS_Q5_K_M.gguf` | 5.5 ГБ | ~7 ГБ | Лучше базового |
| `q4` | `MOSS_TTS_Q4_K_M.gguf` | 4.8 ГБ | ~6 ГБ | Базовое |

> **Рекомендации**: `q8` — лучший баланс качества и скорости для карт с 16 ГБ. `q4` — для карт с 8 ГБ. `f16` — если важно максимальное качество и есть 2 GPU.

### Переключение модели

Используйте скрипт `switch-model.sh`:

```bash
# Интерактивный выбор
bash switch-model.sh

# Переключить напрямую
bash switch-model.sh q8
bash switch-model.sh f16
```

Скрипт показывает текущую модель, список доступных с размерами и обновляет `default.yaml` автоматически. После переключения перезапустите сервер.

Модель также можно выбрать при установке:

```bash
bash setup.sh --hf-token hf_abc123 --model q8
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
| `--seed N` | Random seed по умолчанию |

Примеры запуска:

```bash
# Доступ только локально
python server.py --config configs/llama_cpp/default.yaml

# Доступ из сети
python server.py --config configs/llama_cpp/default.yaml --host 0.0.0.0 --port 8080

# На одной GPU
CUDA_VISIBLE_DEVICES=0 python server.py --config configs/llama_cpp/default.yaml

# Клонирование голоса через reference audio
python server.py --config configs/llama_cpp/default.yaml --reference voice_sample.wav

# Режим экономии памяти (8 ГБ GPU)
CUDA_VISIBLE_DEVICES=0 python server.py --config configs/llama_cpp/trt-8gb.yaml --low-memory
```

### Фиксация голоса

По умолчанию модель генерирует речь **разным голосом** при каждом запросе из-за случайного сэмплирования. Надёжный способ зафиксировать голос — **reference audio** (клонирование):

```bash
python server.py --config configs/llama_cpp/default.yaml --reference voice_sample.wav
```

Передайте WAV-файл с образцом нужного голоса (24 kHz). Модель будет генерировать в этом стиле для всех запросов.

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
| `seed` | int | нет | Random seed (переопределяет серверный `--seed`) |
| `format` | string | нет | Формат ответа: `wav` (по умолчанию) или `raw` |

### Примеры запросов

```bash
# Простая генерация
curl -X POST http://localhost:8000/tts \
    -H "Content-Type: application/json" \
    -d '{"text": "Hello, world!"}' \
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
    json={"text": "Hello from Python!"},
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

Это не ошибка. ONNX Runtime пытается использовать TensorRT, не находит его и переключается на CUDA. Для устранения предупреждения можно установить TensorRT:

```bash
pip install tensorrt==10.*
```

---

## Структура файлов после установки

```
.
├── llama.cpp/                          # Собранный llama.cpp
│   └── build/bin/libllama.so
├── MOSS-TTS/
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
│   └── setup.sh                        # Скрипт автоустановки
│   └── switch-model.sh                 # Скрипт переключения модели
```

---

## Ссылки

- Оригинальный репозиторий: [https://github.com/OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS)
- GGUF веса: [https://huggingface.co/OpenMOSS-Team/MOSS-TTS-GGUF](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-GGUF)
- ONNX аудио-токенизатор: [https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-ONNX)
- llama.cpp: [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
