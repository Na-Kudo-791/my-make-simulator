# 💄 リップメイクシミュレーター（Real-time Lip Makeup Simulator）

衛生面の不安や、似合う色を見つけにくいという課題を解決するために開発した、リアルタイムのリップメイクシミュレーターです。
Streamlit + MediaPipe + OpenCV を使用しています。

---

## ✨ 主な機能

- Webカメラによるリアルタイムな顔検出と高精度なリップトラッキング
- 選択したカラーを唇領域へ自動で合成
- 人気製品をイメージした豊富なカラーバリエーション
- リップカラーの透明度調整（発色の強弱をコントロール）
- リップライナー効果（色・太さ・濃さを自由にカスタマイズ可能）
- リップグロス効果（3本線のハイライトによる自然なツヤ感を表現、強さも調整可能）
- 自然で肌なじみの良いカラー合成

---
## 今後の改善点

- 唇の検出精度向上(マスク補正や追加学習)
- ツヤ、マットなどの質感の再現
- アイシャドウ、チーク、ベースメイクなどへの展開
- スマホ対応/スナップショット保存機能
- ブランドをまたいでのシミュレーションの実装

## 🛠 技術スタック

-  Python 3.9+
- Streamlit
- streamlit-webrtc
- MediaPipe
- OpenCV
- NumPy
- AV

---

## 📦 セットアップ方法

1. 仮想環境の作成（推奨）
```bash
python -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

2.必要ライブラリのインストール
pip install -r requirements.txt

3.アプリの起動
streamlit run app.py
