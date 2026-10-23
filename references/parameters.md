# 完整模型與媒體參數

CLI 用 `--hyphen-name`，manifest 的 `defaults`／`options` 用 `snake_case`。下面列出**本包裝器實際支援**的欄位。程式唯一來源為 `scripts/config.py`。

## 模型與推論

| CLI／JSON | 預設 | 可用值 | 效果與注意事項 |
| --- | --- | --- | --- |
| `--model-version`／`model_version` | `"1.6"` | `"1.6"`、`"1.5"` | 字串，不是JSON小數。自動配對模型與設定；不自動降版 |
| `--inference-steps`／`inference_steps` | 20 | 整數1–100 | 上游建議20–50。10可快速試；較多通常較慢，品質不保證單調變好 |
| `--guidance-scale`／`guidance_scale` | 1.5 | 1.0–5.0 | 上游建議1–3。較大更強調音訊條件，也可能造成抖動；大於3屬額外實驗範圍 |
| `--seed`／`seed` | 1247 | -1或0–4294967295 | -1隨機，實際seed寫入report。固定seed便於比較，仍不保證跨GPU逐位元相同 |
| `--precision`／`precision` | auto | auto、fp16、fp32 | auto在CUDA capability major≥7採fp16，否則fp32。fp32更耗顯存，沒有bf16或int8模式 |
| `--enable-deepcache`／`enable_deepcache` | false | 布林 | DeepCache interval3、branch0固定；加速可能改變品質。關閉用`--no-enable-deepcache` |
| `--vae-slicing`／`vae_slicing` | false | 布林 | 啟用VAE slicing，實際記憶體效果依工作；關閉用`--no-vae-slicing` |

**Profile**：`preview`=1.6／10steps、`balanced`=1.6／20steps、`quality`=1.6／30steps。其他值來自共用預設。顯式欄位會覆蓋profile，包含手動指定1.5；不想降版就保持1.6。

1.6使用 `stage2_512.yaml` 的512人臉區域；1.5使用 `stage2.yaml` 的256人臉區域。上下文固定16影格，不把 `num_frames`、任意解析度或training batch size開放為一般旋鈕，以免破壞已訓練模型的配對。

上游 inference.py 對T4的自動dtype判斷會落到fp32。本工具以獨立adapter明確選dtype，沒有修改／字串替換上游原始碼。主要預設仍是1.6＋A100。

## 裁切、時長與音訊

| CLI／JSON | 預設 | 可用值 | 效果與注意事項 |
| --- | --- | --- | --- |
| `--video-start`／`video_start` | 0秒 | ≥0有限數字 | 從原影片起點裁掉這段；先裁切再循環 |
| `--audio-start`／`audio_start` | 0秒 | ≥0有限數字 | 從替換音檔起點裁掉這段 |
| `--duration`／`duration` | null | ≥0.64秒，或JSON null | 輸出上限，不是強制延長。CLI省略代表不額外限制；CLI不接受0代表自動 |
| `--length-mode`／`length_mode` | shortest | shortest、audio、video | 決定時長與補齊方式，見下表 |
| `--audio-offset`／`audio_offset` | 0秒 | 有限正／負數 | 正數在裁切後音檔前加靜音；負數再裁掉音檔開頭，作用於推論前 |
| `--audio-gain-db`／`audio_gain_db` | 0dB | -60–24 | 替換音訊增益；不做響度標準化／limiter，過大可能削波 |

| 模式 | 目標時長（再套duration上限） | 影片不足 | 音訊不足 |
| --- | --- | --- | --- |
| shortest | 裁切／offset後兩者較短 | 截到較短 | 截到較短 |
| audio | 裁切／offset後音訊長度 | 正向循環裁切後的影片 | 不另延長 |
| video | 裁切後影片長度 | 不另延長 | 尾端補靜音 |

例：影片10秒、音訊6秒，沒有其他裁切時，shortest=6秒、audio=6秒、video=10秒（後4秒靜音）。音訊加 `audio_offset=2` 後，shortest=8秒，前2秒是靜音。正offset會增加可用音訊長度，負offset會減少；若裁掉全部有效來源會報錯。

時長最後向下對齊25fps的完整影格，可能比要求短不到0.04秒。至少需要16影格（0.64秒）。固定25fps不提供任意fps參數，避免Whisper音訊特徵與視訊時間基準不一致。

## 尺寸與長片分段

| CLI／JSON | 預設 | 可用值 | 效果與注意事項 |
| --- | --- | --- | --- |
| `--max-side`／`max_side` | 1280 | 0，或256–4096的偶數 | 整張影片最長邊上限、等比例縮小、不放大；0保留尺寸但仍調整成偶數 |
| `--segment-seconds`／`segment_seconds` | 0 | 0或≥1秒 | 0整支推論；>0切片。切點向下對齊25fps；最後不足0.64秒併入前一段 |
| `--keep-intermediates`／`keep_intermediates` | false | 布林 | 成功後保留遠端工作目錄；關閉用`--no-keep-intermediates`。失敗中間檔保留供診斷 |

模型512指人臉工作區，不是全片輸出大小。1920×1080搭配max_side1280會成1280×720；直式1080×1920會成720×1280。想保留原尺寸可設0，但整支影片影格會占用更多RAM和處理時間。

分段沒有重疊／交叉淡化，不保證無接縫；每段seed為`(effective_seed + 段落索引) % 2**32`。修改segment_seconds會改變結果。最後的音軌由完整正規化音訊重封裝，避免每段AAC累積時間差。

`keep_intermediates`只影響遠端檔案，沒有自動下載中間檔。停止session後遠端檔案會消失；想檢查須同時保留session，手動下載所需檔案。

## 輸出編碼

| CLI／JSON | 預設 | 可用值 | 效果與注意事項 |
| --- | --- | --- | --- |
| `--crf`／`crf` | 18 | 整數0–51 | 最終H.264品質；數值低檔案大。0也不能還原上游／前處理已壓縮細節 |
| `--preset`／`preset` | medium | ultrafast、superfast、veryfast、faster、fast、medium、slow、slower、veryslow | x264編碼速度與壓縮效率；不影響diffusion步數 |
| `--audio-bitrate`／`audio_bitrate` | 192k | 96k、128k、160k、192k、256k、320k | 最終AAC bitrate；音訊仍為16kHz mono，不會恢復原始立體聲／高頻 |

固定輸出MP4、H.264、yuv420p、25fps、AAC、faststart。輸入會正規化並重新編碼，並非無損剪輯。最終CRF只管最後一次編碼，不控制上游模型內部FFmpeg。

## 不支援的選項

沒有prompt、negative prompt、strength、任意fps、face index、多人分別換嘴、聲音複製、原聲混音、臉部修復、背景生成、TensorRT、CPU offload或訓練設定。請勿把其他模型教學的旗標直接加進manifest。

`max_side`是整片縮放，不是crop；若要從多人鏡頭挑一人或放大遠處臉，先另外裁片再輸入。`audio_offset`是同步輸入的前處理，不是SyncNet自動估測偏差。
