# Extract Subtitle

批量生成视频英文字幕的工具，基于 OpenVINO GenAI 和 Whisper 模型。

## 特性

- 使用 OpenVINO GenAI 优化推理性能
- 支持 Whisper Large V3 Turbo 模型（INT4 量化）
- 音频流式处理，直接从视频提取音频到内存，无需临时文件
- 支持单文件和批量处理
- 自动生成标准 SRT 格式字幕

## 依赖

- Python 3.8+
- openvino-genai
- numpy
- ffmpeg

## 安装

```bash
pip install openvino-genai numpy
```

确保 ffmpeg 已安装并在 PATH 中：

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# Arch Linux
sudo pacman -S ffmpeg

# macOS
brew install ffmpeg
```

## 使用方法

### 处理单个视频文件

```bash
python batch_subtitle.py -i video.mp4
```

### 批量处理目录中的视频

```bash
python batch_subtitle.py -d /path/to/videos
```

### 指定模型路径

```bash
python batch_subtitle.py -i video.mp4 -m /path/to/whisper-model
```

### 使用 CPU 推理

```bash
python batch_subtitle.py -i video.mp4 --device CPU
```

## 参数说明

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--input` | `-i` | 单个视频文件路径 | - |
| `--directory` | `-d` | 视频文件目录路径 | - |
| `--model` | `-m` | Whisper 模型路径 | `whisper-large-v3-turbo-int4-ov` |
| `--device` | - | 推理设备 (CPU/GPU) | `GPU` |
| `--keep-wav` | `-w` | 保留临时 WAV 文件（已弃用） | `False` |
| `--extensions` | `-e` | 要处理的视频扩展名 | `mp4,mkv` |

## 支持的视频格式

- MP4
- MKV
- AVI
- MOV
- FLV
- WMV

## 工作原理

1. 使用 ffmpeg 从视频中提取音频（16kHz, 单声道）
2. 音频数据直接流式传输到内存（不落盘）
3. Whisper 模型进行语音识别转录
4. 生成带时间戳的 SRT 字幕文件

## 注意事项

- 当前配置仅支持英语转录（`language = "<|en|>"`）
- 首次运行会加载模型，可能需要几秒钟
- GPU 推理需要支持 OpenVINO 的硬件（如 Intel Arc GPU）

## 许可证

MIT License
