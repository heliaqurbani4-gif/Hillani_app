import streamlit as st
import numpy as np
import pandas as pd
import cv2, os
from PIL import Image
from math import hypot

# ---------- Page config ----------
st.set_page_config(
    page_title="Hillani — Rhinoplasty Photo Measurements",
    page_icon="assets/hillani_logo.png",
    layout="wide"
)

# ---------- Light CSS (mobile-friendly) ----------
css = """
<style>
@media (max-width: 768px) {
  .block-container {padding-top: .5rem; padding-bottom: 2rem;}
  header, footer {visibility: hidden;}
  .stButton>button {width: 100%;}
}
.logo {display:flex; align-items:center; gap:.6rem;}
.logo img {height:36px;}
.muted {color:#6b7480;font-size:.9rem;}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# ---------- Header ----------
c1, c2 = st.columns([1,4])
with c1:
    st.image("assets/hillani_logo.png", use_container_width=False)
with c2:
    st.markdown(
        "<div class='logo'><h2 style='margin:0'>Hillani</h2>"
        "<span class='muted'>Rhinoplasty Photo Measurements</span></div>",
        unsafe_allow_html=True
    )

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Metadata")
    patient_id = st.text_input("Patient_ID", "P01")
    rater_id   = st.selectbox("Rater_ID", ["Auto", "Rater1", "Rater2", "Other"])
    view_hint  = st.selectbox("View (hint)", ["Auto", "Basal", "Frontal", "Lateral"])
    photo_date = st.date_input("Photo Date")
    photo_quality = st.selectbox("Photo Quality", ["OK", "Blur", "Shadow", "Perspective"])
    skin_class = st.selectbox("Skin Thickness", ["", "Thin", "Medium", "Thick"])
    cart_class = st.selectbox("Cartilage Stiffness", ["", "Soft", "Moderate", "Firm"])
    tip_support = st.selectbox("Tip Support", ["", "Weak", "Moderate", "Strong"])
    st.caption("Mobile-friendly • Works great on phones")

# ---------- Upload ----------
st.subheader("1) Upload a photo")
img_file = st.file_uploader("PNG / JPG", type=["png","jpg","jpeg"])

if img_file is None:
    st.info("Upload an image to begin.")
    st.stop()

pil = Image.open(img_file).convert("RGB")
img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
h, w = img.shape[:2]

def distance(p1, p2):
    return float(hypot(p1[0]-p2[0], p1[1]-p2[1]))

# ---------- Optional Auto-detect (only if mediapipe is available) ----------
st.subheader("2) Auto-detect (optional)")
HAS_MP = False
try:
    import importlib
    HAS_MP = importlib.util.find_spec("mediapipe") is not None
except Exception:
    HAS_MP = False

auto = {}

if HAS_MP:
    if st.button("🚀 Auto-detect view + landmarks + measurements"):
        try:
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=1) as fm:
                res = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if not res.multi_face_landmarks:
                st.error("No face landmarks detected. Try a clearer image.")
            else:
                lms = res.multi_face_landmarks[0].landmark
                idx = {"nose_tip":1,"nasion":168,"alar_left":94,"alar_right":326}
                def xy(i):
                    p = lms[i]; return (p.x*w, p.y*h)
                tip = xy(idx["nose_tip"]); nasion = xy(idx["nasion"])
                al_l = xy(idx["alar_left"]); al_r = xy(idx["alar_right"])
                al_mid = ((al_l[0]+al_r[0])/2.0, (al_l[1]+al_r[1])/2.0)

                nasal_len = distance(nasion, tip)
                proj = distance(al_mid, tip)
                goode = proj/nasal_len if nasal_len>0 else None

                overlay = img.copy()
                for p in [tip, nasion, al_l, al_r, al_mid]:
                    cv2.circle(overlay, (int(p[0]),int(p[1])), 4, (0,255,255), -1)
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Auto landmarks", use_container_width=True)
                if goode:
                    st.success(f"Goode ≈ {goode:.3f} (len={nasal_len:.1f}px, proj={proj:.1f}px)")
                auto = dict(View_Auto="Frontal-like", Nasal_Length_px=nasal_len,
                            Tip_Projection_px=proj, Goode_Ratio=goode)
        except Exception as e:
            st.error(f"Auto-detect failed: {e}")
else:
    st.info("Auto-detect is unavailable on this deployment (mediapipe not installed). "
            "You can still upload and save data; auto-mode can be enabled later.")

# Always show the uploaded image
st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Preview", use_container_width=True)

# ---------- Save ----------
st.subheader("3) Save to CSV")
if st.button("💾 Save results"):
    os.makedirs("output", exist_ok=True)
    row = dict(
        Patient_ID=patient_id, Rater_ID=rater_id, View_Hint=view_hint, View_Auto=auto.get("View_Auto",""),
        Photo_Date=str(photo_date), Photo_Quality_Flag=photo_quality,
        Skin_Thickness_Class=skin_class, Cartilage_Stiffness_Class=cart_class, Tip_Support_Class=tip_support,
        Nasal_Length_px=auto.get("Nasal_Length_px"), Tip_Projection_px=auto.get("Tip_Projection_px"),
        Goode_Ratio=auto.get("Goode_Ratio")
    )
    out_path = "output/hillani_measurements.csv"
    if os.path.exists(out_path):
        old = pd.read_csv(out_path)
        df = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(out_path, index=False)
    st.success(f"Saved → {out_path}")
    st.dataframe(df.tail(5), use_container_width=True)

