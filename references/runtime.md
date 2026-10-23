# 執行、狀態與恢復

## 工作如何運作

本機runner只使用Python標準函式庫；有ffprobe時可作媒體檢查。真實推論依賴官方Colab CLI；本機呼叫以argv陣列執行，不經shell拼接素材路徑。

runner上傳共用config與remote_runtime到唯一的 `/content/ls_<run-id>`，用 `colab exec --file` 傳送driver；driver以 `subprocess.run(check=True)` 呼叫遠端程式並在成功後印出唯一標記。**CLI回傳0但沒有成功標記仍算失敗**，因為某些CLI版本會只顯示kernel錯誤卻不回傳非零exit code。

遠端環境在 `/content/latentsync-skill`；以檔案鎖防止同一root並行安裝／推論。採用uv管理的Python3.10、獨立venv、PyTorch2.5.1＋CUDA12.1 wheels、固定上游commit及固定Hugging Face revision。下載Whisper tiny、UNet、VAE與InsightFace必要偵測模型，`pip check`確認宣告相依性。

環境與模型快取在同一runtime可重用；停止／回收runtime後會消失。沒有跨session自動Drive快取。不要同時在同一runtime啟動多個runner。

每個工作產生自己的ASCII工作目錄，將素材正規化後交給上游。這是必要隔離：上游含未引用的shell FFmpeg路徑，也會刪除自己的temp目錄。使用者中文、空格與特殊字元檔名不直接進入該shell命令。

## 狀態的意思

| 狀態 | 意義 |
| --- | --- |
| dry_run | 僅完成本機驗證 |
| setup / uploading / generating / downloading | 工作目前階段 |
| completed | 下載及驗證完成；還要看session_status確認釋放 |
| partial | 至少一項完成，後續有失敗 |
| failed | 沒有工作完成或目前工作失敗 |
| uncertain | 逾時／中斷，不能斷定遠端是否結束 |
| cancelled | 前項失敗，因此後項尚未執行 |
| cleanup_failed / stop_failed | 收尾停止失敗，需要檢查runtime |
| stopped | runner的stop呼叫成功 |
| kept | 依選項保留session |
| unknown_busy | 既有session可能仍在執行；先查明再送工作 |

遠端 `report.json` 記錄模型、GPU、實際seed、片段範圍、時長、輸出雜湊與來源。最後MP4在遠端經完整FFmpeg解碼檢查；本機以ffprobe確認影音stream，若未安裝ffprobe則本機只能驗證檔案／雜湊，請優先安裝ffmpeg套件。

## 逾時與續跑

1. 先看progress中的session名稱、工作狀態與remote_output。
2. 查 `python3 scripts/runner.py status --session NAME`。狀態資訊只是觀測，不保證推論絕對已結束。
3. 自建session正常收尾會停止；既有session未加stop-on-complete時不擅自停止。必要時明確執行 `stop --session NAME`。
4. 確認遠端工作不再重複執行後，同manifest／progress加 `--resume`。完成項目必須仍符合輸入、參數、config／remote程式內容指紋與成品驗證才能略過。程式改版後可能需要重跑。
5. 未完成的模型工作從頭開始。若已有變動過的目的檔，換檔名或明確overwrite。

收到Ctrl+C時會嘗試清理；kill -9、關機或無法連線不能保證清理。停止失敗不代表影片不存在；保留成品與log，再解決runtime。progress已有檔案時預設拒絕重寫，要用resume、新路徑或overwrite。若留下`.lock`，先確認沒有本機／遠端工作使用該紀錄再手動刪除；不可靠刪鎖讓兩個工作同時執行。

未自動重試付費配置或生成工作。CLI自己的底層網路重試依其版本；本工具不透過重複配置繞過額度／供應限制。

## 可重現性與驗證邊界

固定commit、模型revision與seed能記錄大部分條件，不能保證不同GPU／驅動／底層套件位元一致。安裝後的 `environment.freeze.txt` 與 `environment.manifest.json` 留在遠端root，精確套件版本可供排查。Python、uv、pip、建置工具、PyTorch與推論套件的固定版本都納入環境快取指紋，版本修改會建立新venv。直接依賴有固定版本，並非所有傳遞依賴都有lockfile。

本地測試涵蓋參數與manifest驗證、假CLI生命週期、下載失敗、成功標記、續跑與安裝備份；實際FFmpeg測試涵蓋影格率、採樣率、裁切、offset、補齊／循環及輸出可解碼性。這些都不驗證人臉模型品質、Colab帳號權限、GPU供應或遠端CUDA執行。

2026-10-24驗證紀錄：共69項測試。Windows上67通過、2項symlink權限相關測試跳過；Ubuntu WSL上65通過、4項因未安裝FFmpeg跳過。Windows實際FFmpeg測試與WSL實際symlink測試互補。另以真正WSL對中文／空格路徑執行雙工作dry-run，未呼叫Colab。Notebook通過nbformat格式、全部程式cell語法及內嵌原始碼一致性檢查；skill通過quick_validate。

首次真實smoke test建議5秒單人正面片段、1.6、A100、20steps、guidance1.5，記錄成品及report並檢查嘴型、聲音、時長、臉部身份與清理結果。只有完成這一步才可宣稱該帳號環境端到端已驗證。
