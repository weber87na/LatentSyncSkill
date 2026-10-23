---
name: latentsync-colab
description: 使用 WSL 或 Linux 的 Colab CLI，將本機影片與替換音檔交給 Colab GPU 執行 LatentSync 1.6 嘴型同步；支援單支、批次、片段裁切、時長對齊、參數調整、下載與續跑。適用於已有影片及音檔的嘴型重製，不包含配音生成、聲音複製、發布或模型訓練。
---

# LatentSync Colab

以 **WSL + Colab、LatentSync 1.6、A100** 為主要路徑。這是已有影片的嘴型重製，不是從文字產生影片。1.5 僅供使用者明確選擇，GPU 配置失敗或 OOM 不得偷偷降版。

## 入口與選項

- 人工安裝、WSL 路徑、首次授權：讀 [README.md](README.md)。
- 模型、媒體、編碼參數與調整例子：讀 [references/parameters.md](references/parameters.md)。
- 部署、失敗恢復、已知限制：讀 [references/runtime.md](references/runtime.md)。
- 來源版本與驗證邊界：[references/sources.md](references/sources.md)。
- 使用 `python scripts/runner.py --help` 及子命令 `--help` 取得實際介面；以此 skill 目錄為基準找到腳本，不依賴目前工作目錄。

## 執行工作

1. 取得影片、替換音檔與輸出路徑。使用者已給路徑就直接沿用；缺必要素材才詢問。Windows 檔案在 WSL 用 `/mnt/c/...`、`/mnt/d/...`，含空格須引用。
2. `doctor` 檢查本機工具；首次 Google OAuth 請使用者在互動式終端執行 `colab --auth=oauth2 usage`。不讀取或複製 token。正常 dry-run 不接觸帳號、建立 GPU 或執行模型。
3. 準備 `infer` 或 `batch`，先 `--dry-run` 驗證所有輸入／輸出路徑及選項。預設 1.6、20 steps、guidance 1.5、seed1247；首次品質確認可先取3–5秒。長影片視需求使用 `--segment-seconds`，說明接縫可能不連續。
4. 使用者要求執行嘴型同步即已授權該工作所需的上傳與 Colab 推論；只有建立 skill、查選項或 dry-run 的要求不授權建立付費 runtime。一般授權範圍內不額外要求確認。若用途涉及 Colab 禁止的 deepfake，不要以授權或付費方案宣稱例外；查閱服務規範。
5. runner 處理配置、上傳、安裝、推論、下載與驗證。勿把 CLI exit code 0 當成遠端執行成功：runner 另驗證成功標記、MP4 音訊／視訊、report 與雜湊。
6. 回報成品、模型版本、參數、GPU、是否已停止 session。以真實 report 為準；本地驗證、FFmpeg 測試、mock CLI 測試都不等於完成真實 GPU 推論。

執行前檢視代表影格：上游會排除偵測框寬小於50px或高小於80px的人臉，每個影格都要有合格人臉。遠景即使整片為1080p也可能失敗；必要時先建立人物區域的裁切放大副本，保留原片，並避開鏡頭晃動與遮擋。單純縮小整張影片可能讓問題更嚴重。若另選片段、裁切或貼回原構圖，回報實際片段起點與處理方式；目前runner沒有自動追蹤裁切參數。

## Session 與續跑

- runner 自建 session 預設收尾停止；`--keep-session` 僅在確定要重用時指定。既有 `--session` 只有指定 `--stop-on-complete` 才會由 runner 停止。
- 逾時或斷線後不可盲目再開一台 GPU。讀 progress、查 session 狀態；既有 session 可能仍在執行。先確認遠端停止或完成，再續跑。
- `--resume` 只跳過輸入、參數及本機成品仍匹配的完成項目，不恢復 diffusion 中途狀態。不要刪除成功成品來重試。
- 批次順序執行；不要在同一 runtime 並行多個模型工作。`--overwrite` 僅用於使用者授權的輸出替換。

## 實作邊界

1.6 是512人臉區域模型，不表示最終影片固定512×512。輸入正規化25fps，音訊16kHz mono；`max_side` 控制整張影片下採樣，原音由替換音檔取代。

模型、程式、Whisper與VAE使用配對固定版本。不要把任意 `resolution`、訓練 batch size、face index、prompt、strength、臉部修復或 CPU offload 等不存在的參數傳給此 runner。若使用者需要新增功能，先實作與驗證，再對外宣稱支援。
