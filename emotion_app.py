
import streamlit as st
from deepface import DeepFace
from PIL import Image
import tempfile
import matplotlib.pyplot as plt

# 감정별 이모지 매핑
emotion_icons = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐"
}

st.title("감정 인식 웹앱 with 이모지")

uploaded_file = st.file_uploader("사진을 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드한 이미지", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        result = DeepFace.analyze(img_path=tmp_file.name, actions=["emotion"], enforce_detection=False)[0]["emotion"]

    st.subheader("감정 분석 결과")

    # 이모지와 감정 이름 같이 출력
    for emotion, score in result.items():
        icon = emotion_icons.get(emotion, "")
        st.write(f"{icon} **{emotion.capitalize()}** : {score:.2f}%")

    # 차트 그리기
    fig, ax = plt.subplots()
    ax.bar(result.keys(), result.values(), color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel("감정 점수 (%)")
    plt.title("감정별 확률")
    st.pyplot(fig)
