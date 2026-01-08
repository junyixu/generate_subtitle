import argparse
import subprocess
import sys
from pathlib import Path

def extract_audio_from_video(video_path, sample_rate=16000):
    """
    通过 ffmpeg 从视频中提取音频 (PCM s16le)，不落盘
    """
    import numpy as np

    video_path = Path(video_path)
    cmd = [
        'ffmpeg',
        '-hide_banner',
        '-loglevel', 'error',
        '-i', str(video_path),
        '-vn',
        '-ar', str(sample_rate),
        '-ac', '1',  # 单声道
        '-f', 's16le',
        '-acodec', 'pcm_s16le',
        '-',
    ]
    
    try:
        print(f"提取音频: {video_path.name}")
        completed = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audio = np.frombuffer(completed.stdout, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            print("提取失败: 未获得音频数据")
            return None
        return audio
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", errors="replace")
        print(f"提取失败: {stderr.strip()}")
        return None
    except FileNotFoundError:
        print("提取失败: 未找到 ffmpeg，请先安装并确保在 PATH 中")
        return None

def transcribe_audio_to_srt(video_path, model_path="whisper-large-v3-turbo-int4-ov", device="GPU"):
    """
    将视频文件转录为SRT字幕（音频通过管道获取，不落盘）
    """
    try:
        # 初始化模型
        print(f"加载模型: {model_path}")
        pipe = openvino_genai.WhisperPipeline(model_path, device)
        
        # 配置
        config = pipe.get_generation_config()
        config.language = "<|en|>"
        config.task = "transcribe"
        config.return_timestamps = True
        
        # 提取音频（不落盘）
        raw_speech = extract_audio_from_video(video_path, sample_rate=16000)
        if raw_speech is None:
            return None
        
        # 推理
        print("开始转录...")
        result = pipe.generate(raw_speech, config)
        
        # 生成SRT文件
        srt_path = Path(video_path).with_suffix('.srt')
        save_srt(result, srt_path)
        
        print(f"字幕已保存: {srt_path.name}")
        return srt_path
        
    except Exception as e:
        print(f"转录失败: {e}")
        return None

def float_to_srt_time(seconds):
    """
    将秒数转换为SRT时间格式 (00:00:00,000)
    """
    if seconds is None:
        seconds = 0.0
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def save_srt(result, srt_path):
    """
    将转录结果保存为SRT格式
    """
    with open(srt_path, "w", encoding="utf-8") as f:
        if hasattr(result, 'chunks'):
            for i, chunk in enumerate(result.chunks):
                s_time = chunk.start_ts
                e_time = chunk.end_ts
                text_segment = chunk.text.strip()
                
                # 写入序号
                f.write(f"{i+1}\n")
                
                # 写入时间戳
                f.write(f"{float_to_srt_time(s_time)} --> {float_to_srt_time(e_time)}\n")
                
                # 写入文本
                f.write(f"{text_segment}\n\n")
                
                # 打印进度
                print(f"   [{float_to_srt_time(s_time)}] {text_segment[:50]}...")
        else:
            print("未找到时间戳数据")

def process_single_video(video_path, model_path, device, keep_wav=False):
    """
    处理单个视频文件
    """
    print(f"\n处理文件: {video_path}")
    print("-" * 50)

    if keep_wav:
        print("提示: 已启用管道模式，不再生成临时 .wav 文件，`--keep-wav` 将被忽略")
    
    # 转录为 SRT（音频不落盘）
    srt_path = transcribe_audio_to_srt(video_path, model_path, device)
    
    return srt_path is not None

def process_directory(directory, model_path, device, keep_wav=False):
    """
    处理目录中的所有视频文件
    """
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(directory).glob(f'*{ext}'))
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    success_count = 0
    for video_file in video_files:
        if process_single_video(video_file, model_path, device, keep_wav):
            success_count += 1
    
    print(f"\n处理完成！成功: {success_count}/{len(video_files)}")

def main():
    parser = argparse.ArgumentParser(
        description='批量生成视频英文字幕',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  1. 处理单个文件:
     python batch_subtitle.py -i video.mp4
     
  2. 处理整个目录:
     python batch_subtitle.py -d /path/to/videos
     
  3. 指定模型路径:
     python batch_subtitle.py -i video.mp4 -m /path/to/model
     
  4. 保留临时WAV文件:
     python batch_subtitle.py -i video.mp4 -w
        """
    )
    
    # 输入选项（互斥）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input', type=str, 
                           help='单个视频文件路径')
    input_group.add_argument('-d', '--directory', type=str,
                           help='视频文件目录路径')
    
    # 可选参数
    parser.add_argument('-m', '--model', type=str, 
                       default="whisper-large-v3-turbo-int4-ov",
                       help='模型路径 (默认: whisper-large-v3-turbo-int4-ov)')
    parser.add_argument('--device', type=str, default="GPU",
                       choices=['CPU', 'GPU'],
                       help='推理设备 (默认: GPU)')
    parser.add_argument('-w', '--keep-wav', action='store_true',
                       help='保留临时WAV文件')
    parser.add_argument('-e', '--extensions', type=str, default='mp4,mkv',
                       help='要处理的视频扩展名 (逗号分隔) (默认: mp4,mkv)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("批量字幕生成工具")
    print("=" * 60)
    
    try:
        if args.input:
            # 处理单个文件
            video_path = Path(args.input)
            if not video_path.exists():
                print(f"错误: 文件不存在: {args.input}")
                sys.exit(1)
            
            success = process_single_video(
                video_path, args.model, args.device, args.keep_wav
            )
            
            if success:
                print(f"\n✅ 处理完成！字幕已保存到: {video_path.stem}.srt")
            else:
                print(f"\n❌ 处理失败！")
                
        elif args.directory:
            # 处理目录
            directory = Path(args.directory)
            if not directory.exists() or not directory.is_dir():
                print(f"错误: 目录不存在: {args.directory}")
                sys.exit(1)
            
            process_directory(directory, args.model, args.device, args.keep_wav)
    
    except KeyboardInterrupt:
        print("\n\n用户中断操作")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 检查必要的依赖
    try:
        import openvino_genai
        import numpy
    except ImportError as e:
        print(f"错误: 缺少必要的库 - {e}")
        print("请安装以下库:")
        print("  pip install openvino-genai numpy")
        sys.exit(1)
    
    main()

