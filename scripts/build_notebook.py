"""Build a self-contained uploadable Colab notebook from the maintained Python sources."""
from pathlib import Path
import json


ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text, form=False):
    return {"cell_type": "code", "metadata": {"cellView": "form"} if form else {},
            "execution_count": None, "outputs": [], "source": text.splitlines(True)}


def build():
    sources = {name: (ROOT / "scripts" / name).read_text(encoding="utf-8")
               for name in ("config.py", "remote_runtime.py")}
    cells = [markdown("""# LatentSync 1.6 嘴型同步（Colab GPU）

建議主要流程：WSL + `scripts/runner.py`。這份 Notebook 是獨立的手動入口，無須先發布 GitHub 儲存庫。

1. 登入 Google，選 **執行階段 → 變更執行階段類型 → GPU → A100**（L4 亦可嘗試）。1.6 官方最低 18 GB VRAM，T4 不足。
2. 依序執行下方儲存格；首次會下載模型、建立獨立 Python 3.10 環境。
3. 上傳有清楚單人臉部的影片，以及替換的語音音檔。音檔取代原聲。
4. 預覽並下載 MP4 和 report.json，然後 **執行階段 → 中斷連線並刪除執行階段**。

此檔內嵌同版本 Python 原始碼，由 `scripts/build_notebook.py` 產生。修改原始碼後請重新產生 Notebook。
2026-10-24 已以相同 worker 從 WSL CLI 完成 Colab A100 40GB、LatentSync 1.6、FP16 真實推論，成功下載 2.32 秒有聲 MP4並自動停止 session。Notebook已驗證語法與輸出串流，未單獨在網頁介面端到端實跑；請先用短片確認素材品質。
執行會使用 Colab 資源；Colab FAQ 禁止 creating deepfakes，請確認用途符合服務規則。
"""), code("""#@title 1. 展開執行程式（只寫入此 Notebook 的工具目錄）
from pathlib import Path
import json, subprocess, sys, uuid, shutil
SCRIPT_DIR = Path('/content/latentsync-skill-tools')
SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
""" + "SOURCES = " + repr(sources) + "\n" + """for name, source in SOURCES.items():
    (SCRIPT_DIR / name).write_text(source, encoding='utf-8')
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from config import resolve_options
RUNTIME_ROOT = '/content/latentsync-skill'

def run_remote(arguments):
    # Relay child output through Python stdout so Colab displays it in the cell.
    with subprocess.Popen(arguments, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                          errors='replace', bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
        returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, process.args)

print('工具已準備。')
""", True), code("""#@title 2. 模型與媒體參數（先用預設值）
model_version = "1.6" #@param ["1.6", "1.5"]
inference_steps = 20 #@param {type:"integer"}
guidance_scale = 1.5 #@param {type:"number"}
seed = 1247 #@param {type:"integer"}
precision = "auto" #@param ["auto", "fp16", "fp32"]
enable_deepcache = False #@param {type:"boolean"}
vae_slicing = False #@param {type:"boolean"}
video_start = 0.0 #@param {type:"number"}
audio_start = 0.0 #@param {type:"number"}
duration_seconds = 5.0 #@param {type:"number"}
# duration_seconds=0 表示不另設上限；首次建議5秒。
length_mode = "shortest" #@param ["shortest", "audio", "video"]
audio_offset = 0.0 #@param {type:"number"}
audio_gain_db = 0.0 #@param {type:"number"}
max_side = 1280 #@param {type:"integer"}
segment_seconds = 0.0 #@param {type:"number"}
crf = 18 #@param {type:"integer"}
preset = "medium" #@param ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"]
audio_bitrate = "192k" #@param ["96k", "128k", "160k", "192k", "256k", "320k"]
keep_intermediates = False #@param {type:"boolean"}
options = resolve_options({
    'model_version': model_version, 'inference_steps': inference_steps,
    'guidance_scale': guidance_scale, 'seed': seed, 'precision': precision,
    'enable_deepcache': enable_deepcache, 'vae_slicing': vae_slicing,
    'video_start': video_start, 'audio_start': audio_start,
    'duration': None if duration_seconds == 0 else duration_seconds,
    'length_mode': length_mode, 'audio_offset': audio_offset,
    'audio_gain_db': audio_gain_db, 'max_side': max_side,
    'segment_seconds': segment_seconds, 'crf': crf, 'preset': preset,
    'audio_bitrate': audio_bitrate, 'keep_intermediates': keep_intermediates,
})
print(json.dumps(options, ensure_ascii=False, indent=2))
""", True), code("""#@title 3. 安裝環境與下載模型（首次較久；重複執行會檢查快取）
run_remote([sys.executable, str(SCRIPT_DIR / 'remote_runtime.py'),
            'setup', '--root', RUNTIME_ROOT,
            '--model-version', options['model_version']])
""", True), code("""#@title 4. 選擇輸入方式並上傳
input_mode = "upload" #@param ["upload", "paths"]
video_path = "/content/video.mp4" #@param {type:"string"}
audio_path = "/content/audio.wav" #@param {type:"string"}
# paths可使用已上傳檔案，或自行mount後的Google Drive路徑。
from google.colab import files
if input_mode == 'upload':
    print('先選一個影片檔')
    video_upload = files.upload()
    if len(video_upload) != 1:
        raise ValueError('請只選一個影片檔')
    video_path = str(Path(next(iter(video_upload))).resolve())
    del video_upload
    print('再選一個替換音檔（檔名不能與影片相同）')
    audio_upload = files.upload()
    if len(audio_upload) != 1:
        raise ValueError('請只選一個音檔')
    audio_path = str(Path(next(iter(audio_upload))).resolve())
    del audio_upload
if Path(video_path).resolve() == Path(audio_path).resolve():
    raise ValueError('影片與音檔必須是不同路徑')
for path in (video_path, audio_path):
    if not Path(path).is_file() or Path(path).stat().st_size == 0:
        raise ValueError('檔案不存在或空白: ' + path)
print('影片:', video_path, '\\n音檔:', audio_path)
""", True), code("""#@title 5. 執行嘴型同步
OUTPUT = Path('/content/outputs') / ('latentsync16_' + uuid.uuid4().hex[:8] + '.mp4')
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
job_file = SCRIPT_DIR / 'job.json'
job_file.write_text(json.dumps({'video': video_path, 'audio': audio_path,
                               'output': str(OUTPUT), 'options': options}), encoding='utf-8')
run_remote([sys.executable, str(SCRIPT_DIR / 'remote_runtime.py'),
            'run', '--root', RUNTIME_ROOT, '--job', str(job_file)])
if not OUTPUT.is_file() or OUTPUT.stat().st_size == 0:
    raise RuntimeError('未產生成品，請看上方錯誤')
print('完成:', OUTPUT)
""", True), code("""#@title 6. 預覽與下載（完成後記得刪除執行階段）
from IPython.display import Video, display
display(Video(str(OUTPUT), embed=True))
files.download(str(OUTPUT))
report = Path(str(OUTPUT) + '.report.json')
if report.is_file():
    files.download(str(report))
print('請確認MP4已下載，再到「執行階段 → 中斷連線並刪除執行階段」。')
""", True)]
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"name": "LatentSync_1_6_Colab.ipynb", "provenance": []},
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, cell in enumerate(cells):
        cell["id"] = f"latentsync-cell-{i}"
    target = ROOT / "assets" / "LatentSync_1_6_Colab.ipynb"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    build()
