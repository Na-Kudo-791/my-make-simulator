# 💄 リップメイクシミュレーター（Real-time Lip Makeup Simulator）
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)

**「店頭テスターの衛生面への不安」**や**「オンラインでの色選びの難しさ」**といった課題に対し、

Webカメラで手軽に**自分に似合うリップカラーをリアルタイムで試せる**シミュレーターを開発しました。

**いつでも・どこでもパーソナルなメイクアップ体験を提供**することを目指しています。

Streamlit + MediaPipe + OpenCV を使用しています。

---

## ✨ 主な機能と技術的な工夫

-   **リアルタイム顔検出＆高精度リップトラッキング:** Webカメラ映像からMediaPipeを用いて顔を瞬時に検出し、唇のランドマークを正確に追跡します。これにより、動きながらでも自然な色重ねを実現します。
-   **自動カラー合成:** 選択したリップカラーを、検出した唇領域へ高精度に自動合成。唇の形状に合わせて自然にフィットするよう調整しています。
-   **豊富なカラーバリエーション:** 人気製品をイメージした多様なカラープリセットを提供し、ユーザーの選択肢を広げます。
-   **リップカラーの透明度調整:** スライダー操作でリップの発色の強弱をリアルタイムでコントロール。薄付きからしっかり発色まで、好みに合わせた仕上がりをシミュレートできます。
-   **カスタマイズ可能なリップライナー効果:** 唇の輪郭に沿って、色・太さ・濃さを自由に調整できるリップライナー効果を実装。より精細なメイクアップシミュレーションが可能です。
-   **自然なリップグロス効果:** 下唇中央部に3本のハイライト線を生成・合成することで、光沢感のある自然なツヤ感を表現。グロスの強さも調整でき、質感の違いを体験できます。
-   **肌なじみの良いカラー合成:** OpenCVの画像処理技術により、合成後の色が元の肌色や唇の色と自然に馴染むよう工夫しています。

---
## 今後の改善点

このシミュレーターをさらに進化させるため、以下の点に取り組んでいく予定です。

-   **検出精度のさらなる向上:** 口の開閉や表情の変化に対するマスク補正の強化、より高度なトラッキングアルゴリズムの導入、場合によっては追加学習による検出精度の向上を目指します。
-   **多様な質感表現の実現:** ツヤ、マット、メタリックなど、リップ製品ごとの多様な質感の再現アルゴリズムを研究・実装し、よりリアルなシミュレーション体験を提供します。
-   **メイクアップシミュレーションの拡張:** リップだけでなく、アイシャドウ、チーク、ベースメイクなど、顔全体のメイクアップシミュレーションへの展開を検討します。
-   **ユーザー体験の強化:** スマートフォン対応やスナップショット保存機能の実装により、利用シーンを拡大し、ユーザーがシミュレーション結果を共有できるようにします。
-   **ブランド連携の深化:** 実際のブランド製品データや色情報を活用し、ブランドを横断したシミュレーションを可能にするAPI連携やデータ統合を模索します。


## 🛠 使用技術

-   **Python 3.9+:** 主要なロジックを記述するプログラミング言語として採用。
-   **Streamlit:** 迅速なプロトタイピングと直感的なUI構築のために利用。Webカメラ連携も容易でした。
-   **streamlit-webrtc:** WebRTC技術を活用し、ブラウザからのリアルタイムな映像ストリーム処理を実現。
-   **MediaPipe:** 高精度な顔のランドマーク検出とトラッキングを可能にするフレームワークとして、リップ領域の特定に利用。
-   **OpenCV:** 画像の読み込み・処理、色合成、マスキングなど、複雑な画像操作の基盤として活用。
-   **NumPy:** 画像データを効率的に扱うための数値計算ライブラリ。
-   **AV:** `streamlit-webrtc`の内部で映像フレームの処理に使用されるライブラリ。

---

## 📦 セットアップ方法

以下の手順でローカル環境にアプリケーションをセットアップし、起動できます。

1.  **仮想環境の作成（推奨）**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windowsの場合: .\venv\Scripts\activate
    ```

2.  **必要なライブラリのインストール**
    ```bash
    pip install -r requirements.txt
    ```

3.  **アプリケーションの起動**
    ```bash
    streamlit run app.py
    ```
