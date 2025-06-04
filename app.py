import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# リップのランドマークインデックス
OUTER_LIP_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                 375, 321, 405, 314, 17, 84, 181, 91, 146]
INNER_LIP_IDX = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                 324, 318, 402, 317, 14, 87, 178, 88, 95]

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return tuple(reversed(rgb))

def apply_lip_color(image, landmarks, color_bgr, alpha=0.6, apply_gloss=False, gloss_alpha=0.3): # gloss引数を追加
    h, w, _ = image.shape
    # ... (既存のouter_lip_points, inner_lip_points, lip_mask の作成処理はそのまま) ...
    outer_lip_points_list = []
    for i in OUTER_LIP_IDX:
        try:
            outer_lip_points_list.append((int(landmarks[i].x * w), int(landmarks[i].y * h)))
        except IndexError:
            return image
    if not outer_lip_points_list: return image
    outer_lip_points = np.array(outer_lip_points_list, np.int32)

    inner_lip_points_list = []
    for i in INNER_LIP_IDX:
        try:
            inner_lip_points_list.append((int(landmarks[i].x * w), int(landmarks[i].y * h)))
        except IndexError:
            return image
    if not inner_lip_points_list: return image
    inner_lip_points = np.array(inner_lip_points_list, np.int32)

    lip_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(lip_mask, [outer_lip_points], 255)
    cv2.fillPoly(lip_mask, [inner_lip_points], 0)

    # ベースのリップカラー適用
    color_img = np.zeros_like(image)
    color_img[:] = color_bgr
    for c in range(3):
        image[:, :, c] = np.where(
            lip_mask == 255,
            cv2.addWeighted(color_img[:, :, c], alpha, image[:, :, c], 1 - alpha, 0),
            image[:, :, c]
        )


    # グロス効果の適用
    if apply_gloss:
        # ハイライトの位置と形状を定義 (下唇の中央上部を想定)
        # ランドマーク0: 鼻先 (口の左右中心の目安)
        # ランドマーク13: 下唇の上唇側境界の中央
        # ランドマーク17: 下唇の下側境界の中央
        # ランドマーク61, 291: 口角
        try:
            # 楕円の中心座標
            ellipse_center_x = int(landmarks[0].x * w) # 口の左右中心
            # 下唇の上端と下端のY座標から、ハイライトの中心Y座標をやや上寄りに設定
            y_lower_lip_top = landmarks[13].y * h
            y_lower_lip_bottom = landmarks[17].y * h
            ellipse_center_y = int(y_lower_lip_top + (y_lower_lip_bottom - y_lower_lip_top) * 0.25) # 下唇の上から1/4程度の位置

            # 楕円の軸の長さ
            axis_x = int(abs(landmarks[291].x - landmarks[61].x) * w * 0.12) # 口全体の幅の12%程度
            axis_y = int((y_lower_lip_bottom - y_lower_lip_top) * 0.18)     # 下唇の高さの18%程度
            
            if axis_x > 0 and axis_y > 0: # 軸が0以下だとエラーになるためチェック
                gloss_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.ellipse(gloss_mask,
                            (ellipse_center_x, ellipse_center_y),
                            (axis_x, axis_y),
                            0,  # angle
                            0,  # startAngle
                            360, # endAngle
                            255, # color (maskなので白)
                            -1) # thickness (-1で塗りつぶし)

                # グロスマスクを唇の領域のみに限定 (はみ出し防止)
                final_gloss_mask = cv2.bitwise_and(gloss_mask, lip_mask)

                # ハイライト色 (白)
                highlight_color_img = np.zeros_like(image)
                highlight_color_img[:] = (255, 255, 255) # BGRで白

                # ハイライトを適用
                for c in range(3):
                    image[:, :, c] = np.where(
                        final_gloss_mask == 255,
                        cv2.addWeighted(highlight_color_img[:, :, c], gloss_alpha, image[:, :, c], 1 - gloss_alpha, 0),
                        image[:, :, c]
                    )
        except IndexError:
            # ランドマークアクセスでエラーが出た場合はグロス処理をスキップ
            pass # または st.warning("グロス処理でエラー") など

    return image

class LipVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.color_bgr = hex_to_bgr("#D97C87")
        self.alpha = 0.2
        self.apply_gloss = True  # グロス効果のデフォルト状態
        self.gloss_alpha = 0.25 # グロスの強さのデフォルト

    # update_params メソッドを修正
    def update_params(self, color_hex, alpha, apply_gloss, gloss_alpha):
        self.color_bgr = hex_to_bgr(color_hex)
        self.alpha = alpha
        self.apply_gloss = apply_gloss
        self.gloss_alpha = gloss_alpha

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            # apply_lip_color にグロス関連の引数を渡す
            img = apply_lip_color(img, face_landmarks.landmark,
                                  self.color_bgr, self.alpha,
                                  self.apply_gloss, self.gloss_alpha)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("💄 リアルタイム リップメイクシミュレータ (グロス対応版)") # タイトル変更
st.markdown("---")

colors_hex = {
    "アップルソルベ": "#DB232E", "オッドコーラル": "#F95557", "チリンチリン": "#D9414B",
    "ハッシュドチェリー": "#C53D65", "ローザ": "#E73D5D", "ブリュレ": "#AF2C23",
    "コットンピオニー (デフォルト風)": "#D97C87", "ヌードベージュ": "#C8A18F", "ベリーレッド": "#B3003B"
}

st.sidebar.header("🎨 カラー設定")
selected_color_name = st.sidebar.selectbox("カラーを選んでね:", list(colors_hex.keys()), index=6)

st.sidebar.header("✨ 透明度調整")
alpha_value = st.sidebar.slider("リップ色の濃さ (低いほど薄付き):", 0.1, 0.3, 0.2, 0.05) # 最大値を少し上げるかも
st.sidebar.info("ℹ️ リップ色の濃さを調整できます。0.2での使用がおすすめです")

# --- グロス効果のUI追加 ---
st.sidebar.header("💖 グロス効果")
apply_gloss_effect = st.sidebar.checkbox("✨ リップグロス効果をオンにする", value=True)
gloss_alpha_value = st.sidebar.slider(
    "グロスの強さ:",
    min_value=0.0, max_value=0.3, value=0.2, step=0.05,
    disabled=not apply_gloss_effect # グロスオフ時はスライダー無効
)
st.sidebar.info("ℹ️ グロスの強さを調整できます。0.1〜0.2程度での使用がおすすめです。")
# --- UI追加ここまで ---

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

ctx = webrtc_streamer(
    key="realtime-lip-makeup-gloss", # キーを変更
    video_processor_factory=LipVideoProcessor,
    media_stream_constraints={
        "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
        "audio": False
    },
    async_processing=True,
    rtc_configuration=RTC_CONFIGURATION,
)

if ctx.video_processor:
    # video_processor.update_params にグロス関連の引数を渡す
    ctx.video_processor.update_params(
        colors_hex[selected_color_name],
        alpha_value,
        apply_gloss_effect, # チェックボックスの値
        gloss_alpha_value   # スライダーの値
    )
else:
    st.warning("カメラの起動をお待ちください...")

# ... (以降の使い方の説明などはそのまま) ...
st.markdown("---")
st.markdown("### 使い方")
st.markdown("""
1.  **カラー設定** サイドバーから好きな色を選んでください。
2.  **透明度調整** サイドバーのスライダーでリップ色の濃さを調整できます。
3.  **グロス効果** 「リップグロス効果をオンにする」でツヤを出し、「グロスの強さ」で調整できます。
4.  カメラに顔を映すと、リアルタイムで唇に色が適用されます。
""")
st.markdown("※ 照明やカメラの品質によって、色の見え方や検出精度が変わることがあります。")
st.markdown("※ 口を開けた際に、口の中に色がはみ出しにくくなるよう改善しました。")