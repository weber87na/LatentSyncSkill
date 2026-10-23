# LatentSync 1.6 Colab Skill

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weber87na/LatentSyncSkill/blob/main/assets/LatentSync_1_6_Colab.ipynb)

在 **WSL** 準備一段人物影片與一個替換音檔，把推論交給 **Google Colab 的 A100 GPU**，完成後自動把嘴型同步 MP4 下載回電腦。

預設 **LatentSync 1.6**，不會因為 GPU 不足而自動降成 1.5。本機不需要 NVIDIA GPU，也不用安裝 PyTorch、CUDA 或模型權重。Colab 端會建立獨立 Python 3.10 環境；不更換 Notebook 本身的 Python。

此專案參考 [minimax-h3-colab-skill](https://github.com/killkli/minimax-h3-colab-skill) 的本機 runner／遠端 Colab 工作流程，模型採用 [ByteDance LatentSync](https://github.com/bytedance/LatentSync)。runner 與包裝工具為獨立實作。

**驗證狀態：** 已做離線流程測試、實際 FFmpeg 媒體測試、安裝測試及 Notebook 語法檢查；**尚未登入帳戶執行真實 Colab GPU 推論**。首次請先跑下面的 5 秒範例。GPU 分配、模型下載與 CUDA 相容性仍需以首次實跑為準。

## 你需要準備什麼

| 項目 | 需求 |
| --- | --- |
| 電腦環境 | WSL2 的 Ubuntu／Linux；以下終端指令都在 WSL 執行 |
| 本機 Python | runner 需要 Python 3.10+；Colab CLI 使用獨立 Python 3.12 |
| Google 帳號 | 能取得 Colab GPU，且有足夠 compute units／對應方案權限 |
| GPU | 預設 A100；可選 L4。1.6 官方最低 18 GB VRAM，T4 的 16 GB 不足 |
| 影片 | 建議 MP4、單一清楚人臉、嘴部沒有遮擋；最終至少 0.64 秒 |
| 替換音檔 | WAV、MP3、M4A 等 FFmpeg 能解碼且含音訊串流的檔案；建議乾淨人聲 |
| 網路與空間 | 會上傳完整輸入檔案、下載數 GB 模型到 Colab；保留充足遠端磁碟空間 |

Colab CLI 官方目前支援 Linux／macOS，這也是本專案選 WSL 的原因。**不要在 Windows PowerShell 直接跑真實 Colab CLI 推論。** `--high-mem` 是系統 RAM，不是增加 GPU 顯存；A100 供應及費率依帳戶當下情況而定，本工具不保證分配成功或固定價格。

輸出會用替換音檔取代原影片聲音，不自動混回背景音樂；不包含 TTS、聲音複製、字幕生成或發布影片。

## 最短路徑：第一次照著做

### 1. 開啟 WSL，進入本專案

第一次取得專案，在 WSL 執行：

```bash
git clone https://github.com/weber87na/LatentSyncSkill.git
cd LatentSyncSkill
```

若還沒安裝 Git，可先執行 `sudo apt update` 與 `sudo apt install -y git`。

如果已將專案放在 Windows 的 `D:\LatentSyncSkill`，直接進入對應的 WSL 路徑，不必再 clone：

```bash
cd /mnt/d/LatentSyncSkill
```

其他位置自行換成專案所在路徑。先確認看得到 `SKILL.md`、`scripts/`、`assets/`：

```bash
ls
```

### 2. 安裝本機小工具

Ubuntu／Debian WSL 首次安裝：

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ffmpeg curl
```

安裝 uv（若已有可跳過）：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

安裝固定版本的官方 Colab CLI。它用自己的 Python，不影響你的專案環境：

```bash
uv python install 3.12
uv tool install --python 3.12 'google-colab-cli==0.7.4'
colab version
python3 scripts/runner.py doctor
```

如果 `colab: command not found`，重新開啟 WSL 終端，或再次執行上面的 `export PATH=...`。此專案依照 CLI 0.7.4 的介面實作；不要在排錯過程中無條件升級所有套件。

### 3. 安裝 skill（只需一次）

```bash
python3 scripts/install.py
```

預設安裝到 WSL 的 `~/.codex/skills/latentsync-colab`；若有設定 `CODEX_HOME`，使用 `$CODEX_HOME/skills/latentsync-colab`。之後在使用這個 Codex 設定目錄的新對話呼叫 `$latentsync-colab`。

這一步讓 Codex 發現 skill；手動執行 runner 不必安裝 skill。若你使用的是 Windows 原生 Codex，其技能目錄與 WSL 分開；可在 PowerShell 的專案資料夾執行 `python scripts/install.py` 安裝到 Windows，再讓該 skill 透過 WSL 跑推論。不要把兩個環境的路徑混用。

更新既有安裝：

```bash
python3 scripts/install.py --force
```

預設不覆蓋現有版本；`--force` 會先備份再更新。指定其他 **skills 目錄**（安裝器會在裡面建立 `latentsync-colab`）：

```bash
python3 scripts/install.py --dest "$HOME/my-codex/skills"
```

也提供 `bash install.sh` 與 `install.ps1` 啟動器。

### 4. 完成 Google 授權

在你能互動輸入的 WSL 終端執行：

```bash
colab --auth=oauth2 usage
```

依終端印出的網址、驗證碼與登入提示完成授權；WSL 若不能自動開啟瀏覽器，就把網址貼到 Windows 瀏覽器。授權後應看到餘額、每小時用量與 active assignments。此命令只查詢，不會建立 GPU。

runner 預設 `oauth2`。不需把 Google 密碼、token 或 cookies 寫進專案，也不需用 `colab auth` 代替此步驟（後者主要處理 runtime 的 GCP 授權）。已設定 ADC 的進階使用者可在每個 runner 子命令加 `--auth adc`。

### 5. 放好影片與音檔

範例使用：

```text
D:\LatentSyncSkill\inputs\video.mp4
D:\LatentSyncSkill\inputs\speech.wav
```

WSL 建立資料夾：

```bash
mkdir -p inputs outputs
```

用 Windows 檔案總管把實際檔案放進去。也可直接使用其他位置，不必複製到專案；例如 `"/mnt/d/我的影片/講者.mp4"`。路徑含空格或中文時保留雙引號。

### 6. 先驗證，不用 GPU

```bash
python3 scripts/runner.py infer \
  --video "inputs/video.mp4" \
  --audio "inputs/speech.wav" \
  --output "outputs/first-5s.mp4" \
  --duration 5 \
  --progress "outputs/first-5s.progress.json" \
  --dry-run
```

看到 `"status": "dry_run"`、`"model_version": "1.6"`、`"gpu": "A100"` 即表示本機設定驗證通過。dry-run 不登入、不上傳、不配置 GPU；它不能證明人臉可被模型偵測或結果品質合格。

### 7. 正式執行同一個 5 秒工作

把上一個命令的 `--dry-run` 拿掉：

```bash
python3 scripts/runner.py infer \
  --video "inputs/video.mp4" \
  --audio "inputs/speech.wav" \
  --output "outputs/first-5s.mp4" \
  --duration 5 \
  --progress "outputs/first-5s.progress.json"
```

流程依序為：配置 A100 → 安裝遠端環境／下載模型 → 上傳 → 前處理 → 嘴型同步 → 下載／驗證 → 停止自建 session。

首次安裝與下載可能占大部分時間。終端會顯示輸出，**沒有保證幾分鐘完成**。不要因為暫時沒有進度就重複開新工作。

完成後查看：

```text
outputs/first-5s.mp4                 成品
outputs/first-5s.mp4.report.json     實際GPU、seed、版本、參數、時間與雜湊
outputs/first-5s.progress.json       本機流程與session狀態
outputs/first-5s.progress.log        操作紀錄
```

成功結果應有 `status: completed`，正常自建 session 應有 `session_status: stopped`。若是 `cleanup_failed`／`stop_failed`，影片可能已完成，但要另外停止 runtime。

### 8. 跑完整影片

移除 `--duration 5`，換新的輸出和 progress 檔名：

```bash
python3 scripts/runner.py infer \
  --video "inputs/video.mp4" \
  --audio "inputs/speech.wav" \
  --output "outputs/full.mp4" \
  --progress "outputs/full.progress.json"
```

預設取影片與音檔中較短的時長。長素材可加 `--segment-seconds 10` 控制每段處理的記憶體，但段落交界可能有視覺跳動。先用短片確認品質再處理長片。

## 直接叫 Codex 使用

安裝後可貼這段，把路徑改成自己的：

```text
使用 $latentsync-colab，在 WSL 用 Colab A100 和 LatentSync 1.6，
把 /mnt/d/素材/video.mp4 的嘴型同步到 /mnt/d/素材/speech.wav，
輸出 /mnt/d/成品/result.mp4。先做本機 dry-run，通過後直接執行；
完成後下載成品並停止這次新建的 session。
```

只想確認設定、不使用 GPU：

```text
使用 $latentsync-colab 幫我準備批次 manifest，只做 dry-run，不建立 Colab runtime。
```

## 常用設定可以直接抄

以下命令假設目前位於專案根目錄，且 `inputs/video.mp4`、`inputs/speech.wav` 已存在。

**更仔細的 1.6 推論：**

```bash
python3 scripts/runner.py infer --video inputs/video.mp4 --audio inputs/speech.wav \
  --output outputs/quality.mp4 --profile quality --guidance-scale 1.5 \
  --progress outputs/quality.progress.json
```

`quality` 是 30 steps，不保證比 20 steps 更好；先比較口型與臉部穩定度。`preview` 是 10 steps，`balanced` 是 20 steps，三者都是 1.6。

**只處理影片第 12 秒起、音檔第 3 秒起的最多 8 秒：**

```bash
python3 scripts/runner.py infer --video inputs/video.mp4 --audio inputs/speech.wav \
  --output outputs/clip.mp4 --video-start 12 --audio-start 3 --duration 8 \
  --progress outputs/clip.progress.json
```

**音檔較長，讓影片正向循環到音檔結束：**

```bash
python3 scripts/runner.py infer --video inputs/video.mp4 --audio inputs/speech.wav \
  --output outputs/audio-length.mp4 --length-mode audio --segment-seconds 10 \
  --progress outputs/audio-length.progress.json
```

**影片較長，補靜音讓輸出跟影片一樣長：**

```bash
python3 scripts/runner.py infer --video inputs/video.mp4 --audio inputs/speech.wav \
  --output outputs/video-length.mp4 --length-mode video \
  --progress outputs/video-length.progress.json
```

**音訊延後 0.2 秒，音量降低 3 dB：**

```bash
python3 scripts/runner.py infer --video inputs/video.mp4 --audio inputs/speech.wav \
  --output outputs/offset.mp4 --audio-offset 0.2 --audio-gain-db -3 \
  --progress outputs/offset.progress.json
```

這會先改好音訊，再用該音訊生成口型；不是把已生成嘴型的成品再移動聲音。

**1.6 嘗試 L4、分段、VAE slicing：**

```bash
python3 scripts/runner.py infer --video inputs/video.mp4 --audio inputs/speech.wav \
  --output outputs/l4.mp4 --gpu L4 --segment-seconds 5 --vae-slicing \
  --progress outputs/l4.progress.json
```

L4／分段／VAE slicing 都不保證不 OOM。維持 1.6 的前提下，先縮短測試、降低 `max_side` 或使用 A100。降低 inference steps 主要減少時間，通常無法明顯降低單步峰值顯存。

## 批次處理

多支影片請放在同一批次，共用一個 session 的套件與模型磁碟快取。**工作順序執行；每個工作會重新載入模型**，同一工作內的分段才重用已載入 pipeline。

建立 UTF-8 `jobs.json`：

```json
{
  "defaults": {
    "model_version": "1.6",
    "inference_steps": 20,
    "guidance_scale": 1.5,
    "seed": 1247
  },
  "jobs": [
    {
      "id": "intro",
      "video": "inputs/video.mp4",
      "audio": "inputs/speech.wav",
      "output": "outputs/intro.mp4",
      "options": {"duration": 5}
    },
    {
      "id": "long",
      "video": "inputs/video2.mp4",
      "audio": "inputs/speech2.wav",
      "output": "outputs/long.mp4",
      "options": {"guidance_scale": 2.0, "segment_seconds": 10}
    }
  ]
}
```

`video`、`audio`、`output` 的相對路徑以 **manifest 所在資料夾** 為基準。每個 `id` 唯一且只用英文、數字、`_`、`-`；檔案路徑可以中文。JSON 不支援註解或最後多一個逗號。

```bash
python3 scripts/runner.py batch --manifest jobs.json \
  --progress outputs/batch.progress.json --dry-run

python3 scripts/runner.py batch --manifest jobs.json \
  --progress outputs/batch.progress.json
```

另附可編輯的 [examples/jobs.json](examples/jobs.json)。每個工作可省略 `output`，此時預設為 manifest 同層 `outputs/<id>.mp4`；用 `--output-dir` 指定其他目錄（這個 CLI 路徑相對於目前終端工作目錄）。

選項套用順序，由低到高：**內建預設 → profile → manifest defaults → 工作 options → 明確 CLI 選項**。每個工作可有 `profile`；若 defaults 已寫 `inference_steps`，它會覆蓋 profile 的步數。未知欄位或不合範圍的值會報錯，不會默默忽略。

## Session、續跑與停止

預設每次命令建立一個 session，正常完成或失敗後嘗試停止它。**不用自己先 `colab new`。**

需要重用已存在的 session：

```bash
colab --auth=oauth2 new --session lipsync-work --gpu A100
python3 scripts/runner.py infer --video inputs/video.mp4 --audio inputs/speech.wav \
  --output outputs/reused.mp4 --session lipsync-work \
  --progress outputs/reused.progress.json --stop-on-complete
```

`--session` 表示該 session 已存在，不會建立它；GPU 沿用遠端實際裝置，`--gpu` 不會更換。省略 `--stop-on-complete` 就保留既有 session；指定後即使工作失敗也嘗試停止。

讓 runner **新建** 的 session 完成後留下，可加 `--keep-session`，在 progress 找到名稱後供下次 `--session` 使用。計費可能持續。發生逾時／中斷時仍會嘗試停止自己建立的 session，以免留下不確定狀態。

查詢與手動停止：

```bash
python3 scripts/runner.py usage
python3 scripts/runner.py status
python3 scripts/runner.py status --session lipsync-work
python3 scripts/runner.py stop --session lipsync-work
```

批次第二支失敗時，第一支成品會保留。確認沒有還在執行的舊工作後，用 **相同的 progress** 續跑：

```bash
python3 scripts/runner.py batch --manifest jobs.json \
  --progress outputs/batch.progress.json --resume
```

`--resume` 只跳過輸入內容、參數及已驗證成品都匹配的完成工作；**不能接續 diffusion 到一半的步數**。不完整工作會整支重跑。現有成品若已變動／設定不同，預設拒絕覆蓋；要保留舊結果就換新檔名，確定替換才加 `--overwrite`。

每組工作使用不同 progress 路徑。已存在的 progress 需要 `--resume`、換新路徑或明確 `--overwrite` 才會繼續。runner 會建立同名 `.lock` 防止同一 progress 並行寫入；強制關機可能留下 lock，**確認舊程序與遠端工作已停止後**才移除那個 lock 檔。不要對同一 session 同時開兩個 runner。強制關機、殺掉程序或網路斷線，可能無法跑完清理；重新連線後請查 `status`。詳細說明見 [執行與恢復](references/runtime.md)。

## 完整選項

所有模型／媒體／編碼選項、預設值、範圍與副作用都列在 [參數手冊](references/parameters.md)。

```bash
python3 scripts/runner.py --help
python3 scripts/runner.py infer --help
python3 scripts/runner.py batch --help
```

| 工作管理選項 | 預設 | 作用 |
| --- | --- | --- |
| `--video` / `--audio` / `--output` | 必填 | 單支工作的素材與 MP4 成品位置 |
| `--manifest` | batch 必填 | UTF-8 JSON 批次檔 |
| `--output-dir` | manifest 同層 outputs | batch 未指定 output 的工作目的地 |
| `--profile` | balanced | preview10／balanced20／quality30 steps；全部 1.6 |
| `--gpu` | A100 | T4／L4／A100／H100／G4；可用性依帳戶。1.6＋T4 事先拒絕 |
| `--high-mem` | 關閉 | 向 Colab 要求較高系統 RAM；不增加 GPU VRAM |
| `--auth` | oauth2 | oauth2／adc；環境變數 COLAB_AUTH 可設預設 |
| `--colab-bin` | PATH 的 colab | 指定 CLI 執行檔；環境變數 COLAB_BIN 可設預設 |
| `--session` | 新建 | 使用既有 session；不代表新 session 的命名 |
| `--keep-session` | 關閉 | 保留新建 session；逾時／中斷仍嘗試停止 |
| `--stop-on-complete` | 關閉 | 既有 session 在收尾時停止，包含失敗 |
| `--timeout` | 7200 秒 | 每個推論工作的執行上限；非整批總時間 |
| `--setup-timeout` | 3600 秒 | 配置／環境安裝等操作的上限 |
| `--transfer-timeout` | 900 秒 | 每次上傳／下載上限 |
| `--remote-root` | /content/latentsync-skill | 同 runtime 內的環境與模型快取；不是 Drive 永久儲存 |
| `--progress` | outputs/latentsync-progress.json | 本機紀錄，建議每批指定獨立檔名 |
| `--log` | progress 副檔名換 .log | 本機診斷紀錄 |
| `--dry-run` | 關閉 | 本機驗證，零 GPU／上傳／授權呼叫 |
| `--resume` | 關閉 | 驗證完成工作後略過，其餘整支重跑 |
| `--overwrite` | 關閉 | 成功生成並下載後允許替換目的成品及 report |

`doctor --check-auth` 會另外查詢帳號用量，不配置 GPU；請先完成互動式授權。

## 手動 Colab Notebook（備用）

1. 點 [直接在 Colab 開啟](https://colab.research.google.com/github/weber87na/LatentSyncSkill/blob/main/assets/LatentSync_1_6_Colab.ipynb)。
2. 也可開啟 [Google Colab](https://colab.research.google.com/)，選「檔案 → 上傳筆記本」，上傳 [assets/LatentSync_1_6_Colab.ipynb](assets/LatentSync_1_6_Colab.ipynb)。
3. 選 A100 GPU；依序執行六個步驟。
4. 表單可設定所有模型、前處理、輸出參數。預設只跑 5 秒，`duration_seconds=0` 表示不另設時長上限。
5. 下載 MP4 與 report，再手動中斷連線並刪除執行階段。Notebook 不會自動停止你正在操作的 runtime。

Notebook 內嵌必要 Python 程式，可使用上方連結或直接上傳本機檔案，無須在 Colab 另外 clone 本專案。`paths` 模式可讀取已上傳的檔案；如需 Drive，自己掛載後填 `/content/drive/MyDrive/...`。本版不自動掛載 Drive或把模型快取搬到 Drive。

## 常見問題

| 現象 | 怎麼處理 |
| --- | --- |
| `D:\...` 找不到 | WSL 改成 `/mnt/d/...`；路徑加雙引號 |
| Python 太舊 | runner 至少 3.10；可用 `uv run --python 3.12 python scripts/runner.py ...` |
| OAuth 失敗／非互動輸入錯誤 | 在你自己的終端先跑 `colab --auth=oauth2 usage`，完成登入再跑 runner |
| A100 分配失敗 | 查 `usage`、方案與當下供應；可明確改 `--gpu L4`，不會自動降到 T4／1.5 |
| T4 顯存不足 | 1.6 需要更大的 GPU；不要把 high-mem 當成 GPU 顯存 |
| `CUDA out of memory` | 先做5秒、`--segment-seconds 5`、降低 `--max-side`、試 `--vae-slicing` 或 A100 |
| 很長的影片吃滿 RAM | 使用分段與縮小 max_side；上游會讀入片段全部影格，長片成本不能只看VRAM |
| 沒偵測到臉／處理到別人 | 換清楚單人鏡頭，先裁掉其他人。上游選最大合格人臉；沒有 face-index 參數 |
| 嘴型不夠跟音 | 先用乾淨人聲；試 guidance1.5→2.0；固定seed比較 |
| 臉部抖動或變形 | 降低guidance；關閉DeepCache；減少遮擋及側臉；增加steps不保證改善 |
| 輸出少於預期長度 | shortest取較短者；改 audio或video；裁切與offset會改可用時長 |
| 音樂不見／變單聲道 | 本版用16kHz mono替換音軌輸出，沒有原聲混音或原始立體聲保留 |
| 分段交界跳動 | 這是獨立片段推論的限制；增加segment秒數或設0，需更多記憶體 |
| `Output/report already exists` | 續跑用相同progress＋resume；新實驗換檔名；確定替換才overwrite |
| 逾時或缺少成功標記 | 查看log／session；遠端可能仍在跑，先停止或確認結束，別直接重送 |
| `stop_failed` | 用progress中的session名稱執行stop，或在Colab介面管理工作階段 |
| Hugging Face／GitHub下載失敗 | 看網路錯誤；保留原session時可重用快取。模型公開下載通常不需token |
| CUDA／ONNX／套件相依衝突 | 保留log、report；不要改Colab kernel套件硬解。可改新的remote-root重建隔離環境 |

## 使用範圍與第三方條款

Colab 的 [FAQ](https://research.google.com/colaboratory/faq.html) 明列 managed runtimes 禁止「creating deepfakes」；付費方案或人物授權不代表這項服務限制的例外。請確認實際用途符合 Colab 規範；本工具不會規避資源限制、維持假活動或暴露公開推論服務。

LatentSync 程式、模型、VAE 與 InsightFace 偵測模型的授權不相同；模型下載成功不表示取得所有商用權利。版本、來源與授權入口見 [sources.md](references/sources.md)。本儲存庫不包含模型權重、Google 憑證或成品。

## 開發者驗證與目錄

```bash
python3 -m unittest discover -s tests -v
python3 scripts/build_notebook.py
python3 scripts/runner.py infer --help
```

測試不會配置 Colab GPU、不會登入或花 compute units；部分測試使用本機 FFmpeg 產生合成片段。Notebook 產生器把維護中的 `config.py`、`remote_runtime.py` 內嵌進去，修改後記得重新產生。

```text
SKILL.md                    給Codex讀的入口
README.md                   本文件（WSL優先）
agents/openai.yaml          skill介面資料
scripts/config.py           共用選項與嚴格驗證
scripts/runner.py           本機Colab流程、下載、狀態與續跑
scripts/remote_runtime.py   Colab安裝、模型、前處理與推論
scripts/install.py          安裝與備份
scripts/build_notebook.py   重建獨立Notebook
assets/LatentSync_1_6_Colab.ipynb
examples/jobs.json          批次範例
references/                 參數、恢復、來源說明
tests/                      不使用GPU的測試
```
