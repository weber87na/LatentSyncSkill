# 來源與版本

查核日期：2026-10-24（Asia/Taipei）。上游可能改版；本專案的runner針對以下固定來源實作，不使用浮動main取得模型行為。

| 項目 | 固定值／來源 |
| --- | --- |
| LatentSync code | `a229c3948406bc2cf6eaf4873e662e70c6a04746` |
| ByteDance/LatentSync-1.6 | `c42c7e6c8e9c213626389fa7d9a3c444b8536353` |
| ByteDance/LatentSync-1.5（手動選項） | `32a20d29aead0498e3e885e90dbbe8027da1b61b` |
| stabilityai/sd-vae-ft-mse | `31f26fdeee1355a5c34592e401dd41e45d25a493` |
| Colab CLI介面 | google-colab-cli0.7.4；安裝指令固定此版 |
| InsightFace模型 | 官方release v0.7的buffalo_l.zip；只取det_10g.onnx與2d106det.onnx，下載後記錄SHA256 |

- [LatentSync 官方 README](https://github.com/bytedance/LatentSync)：1.5最低8GB、1.6最低18GB；步數與guidance建議。
- [固定版本 inference.py](https://github.com/bytedance/LatentSync/blob/a229c3948406bc2cf6eaf4873e662e70c6a04746/scripts/inference.py)：支援旗標、模型元件、dtype及DeepCache流程。
- [固定版本 pipeline](https://github.com/bytedance/LatentSync/blob/a229c3948406bc2cf6eaf4873e662e70c6a04746/latentsync/pipelines/lipsync_pipeline.py)：25fps、16frame、音訊／影片時間處理與FFmpeg輸出。
- [1.6模型卡](https://huggingface.co/ByteDance/LatentSync-1.6)、[1.5模型卡](https://huggingface.co/ByteDance/LatentSync-1.5)、[VAE模型卡](https://huggingface.co/stabilityai/sd-vae-ft-mse)：權重及各自授權入口。
- [InsightFace Python package](https://github.com/deepinsight/insightface/tree/master/python-package)：程式與預訓練模型條款須分開看；buffalo_l不是LatentSync程式Apache授權的一部分。
- [官方Colab CLI](https://github.com/googlecolab/google-colab-cli)、[PyPI0.7.4](https://pypi.org/project/google-colab-cli/0.7.4/)：Python3.12及Linux/macOS支援、OAuth、session、exec、upload/download。
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)：資源可用性、使用限制與managed runtime規範。
- [參考skill](https://github.com/killkli/minimax-h3-colab-skill)：工作流程概念參考；未複製其runner實作或套用未提供的授權。

LatentSync程式採Apache2.0；其模型卡標示openrail++。InsightFace預訓練模型另有條款，不能因程式開源就推論所有權重可無條件商用。若要商用或再分發，逐項閱讀來源授權；本包不重新授權第三方模型。

模型版本固定、face模型雜湊記錄有助追查，但並非已建立獨立可信的所有下載檔SHA256白名單。首次模型下載仍信任上游服務。
