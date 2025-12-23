import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import noisereduce as nr
import io
import soundfile as sf
import matplotlib.pyplot as plt
import os

# --- KERAS IMPORTS (Modeli yeniden inşa etmek için) ---
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.regularizers import l2

# ============================================================================
# 1. AYARLAR VE SABİTLER
# ============================================================================
st.set_page_config(page_title="Sesle Kimlik Doğrulama", page_icon="🎙️", layout="wide")

SR = 16000
N_MFCC = 40
HOP_LENGTH = 512
MAX_LEN = 128  # 4 saniye için CNN giriş boyutu

# ============================================================================
# 2. MODEL MİMARİSİ VE YÜKLEME FONKSİYONLARI
# ============================================================================

# --- Özel Fonksiyonlar (Custom Layers/Loss) ---
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1.0
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# --- Temel Ağ (Base Network) ---
def build_base_network(input_shape):
    input_layer = layers.Input(shape=input_shape)
    reg = l2(0.01)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation=None)(x)
    
    # Lambda katmanı (L2 Normalize)
    x = layers.Lambda(lambda x: K.l2_normalize(x, axis=1))(x)
    
    return models.Model(input_layer, x, name="Shared_Encoder")

# --- Modeli Yükleyen Ana Fonksiyon (CACHE ile) ---
@st.cache_resource
def load_siamese_model_weights():
    # 1. Model yolunu bul
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # GitHub'da 'model' klasörü içindeyse:
    model_path = os.path.join(current_dir, "model", "siamese_model.h5")
    
    # Dosya kontrolü
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model dosyası bulunamadı: {model_path}")
        return None

    try:
        # 2. Mimarisi Sıfırdan İnşa Et
        input_shape_model = (N_MFCC, MAX_LEN, 1)
        base_network = build_base_network(input_shape_model)

        input_a = layers.Input(shape=input_shape_model, name="Left_Audio")
        input_b = layers.Input(shape=input_shape_model, name="Right_Audio")

        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        distance = layers.Lambda(euclidean_distance, output_shape=eucl_dist_output_shape, name="Euclidean_Distance")([processed_a, processed_b])

        model = models.Model(inputs=[input_a, input_b], outputs=distance)

        # 3. Ağırlıkları Yükle (.h5 dosyasından)
        # Not: load_weights, tam model dosyası olsa bile ağırlıkları çekebilir.
        model.load_weights(model_path)
        
        # Derleme (Compile) - Tahmin için şart değil ama iyi pratik
        model.compile(loss=contrastive_loss, optimizer='adam')
        
        print("✅ Model mimarisi oluşturuldu ve ağırlıklar yüklendi.")
        return model

    except Exception as e:
        st.error(f"❌ Model oluşturulurken hata: {e}")
        return None

# Modeli başlat
model = load_siamese_model_weights()

# ============================================================================
# 3. SES ÖN İŞLEME (Preprocessing)
# ============================================================================

def preprocess_audio_pipeline(audio_bytes, target_sr=SR, fixed_length_samples=SR*4): # 4 Saniye
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes))
        
        # 1. Resample & Mono
        if sr != target_sr:
            y = librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)
        if len(y.shape) > 1:
             y = librosa.to_mono(y)
        
        # 2. Gürültü Azaltma
        try:
            y = nr.reduce_noise(y=y, sr=target_sr)
        except:
            pass # Çok kısa seslerde hata verirse geç

        # 3. Sessizlik Silme
        y, _ = librosa.effects.trim(y, top_db=20)

        # 4. Sabit Uzunluk (4 sn / ~64000 samples)
        # Sizin kodunuzdaki mantık: MFCC boyutunu 128'e denk getirmek.
        # Burada önce sesi padding yapıyoruz.
        target_length = int(SR * 4.0) # 64000
        if len(y) > target_length:
             y = y[:target_length]
        else:
             y = np.pad(y, (0, target_length - len(y)), mode='constant')
            
        # 5. Normalizasyon
        y = librosa.util.normalize(y)

        return y
    except Exception as e:
        st.error(f"Ses işleme hatası: {e}")
        return None

def extract_features_mfcc(y, sr=SR):
    """
    Sizin kodunuzdaki extract_mfcc mantığıyla birebir aynı.
    Çıktı: (1, 40, 128, 1)
    """
    try:
        mfcc = librosa.feature.mfcc(
            y=y, 
            sr=sr, 
            n_mfcc=N_MFCC, 
            hop_length=HOP_LENGTH
        )
        
        # Boyutlandırma (128 width)
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_LEN]
            
        # Model girişi için şekillendirme: (Batch, Height, Width, Channels)
        mfcc_reshaped = mfcc.reshape(1, N_MFCC, MAX_LEN, 1)
        return mfcc_reshaped, mfcc
    except Exception as e:
        st.error(f"MFCC hatası: {e}")
        return None, None

# ============================================================================
# 4. ARAYÜZ (UI)
# ============================================================================

st.title("🛡️ Sesle Kimlik Doğrulama Sistemi")
st.markdown("Siyam Ağı (Siamese CNN) kullanılarak geliştirilmiştir.")

# Yan Menü Ayarları
st.sidebar.header("Ayarlar")
THRESHOLD = st.sidebar.slider("Karar Eşik Değeri (Threshold)", 0.0, 1.0, 0.1100, 0.001)
st.sidebar.info(f"Seçili Eşik: {THRESHOLD}")

col1, col2 = st.columns(2)

# SOL KOLON: REFERANS (ANCHOR)
with col1:
    st.header("1. Referans Ses")
    anchor_file = st.file_uploader("Yetkili Sesi Yükle", type=["wav", "mp3"], key="anchor")
    
    feat_ref = None
    if anchor_file:
        st.audio(anchor_file)
        y_ref = preprocess_audio_pipeline(anchor_file.getvalue())
        if y_ref is not None:
            feat_ref, viz_ref = extract_features_mfcc(y_ref)
            st.success("Referans işlendi.")

# SAĞ KOLON: TEST (INPUT)
with col2:
    st.header("2. Test Sesi")
    # Hem dosya yükleme hem mikrofon seçeneği
    input_method = st.radio("Giriş Yöntemi:", ["Dosya Yükle", "Mikrofon"])
    
    test_file = None
    if input_method == "Dosya Yükle":
        test_file = st.file_uploader("Şüpheli Sesi Yükle", type=["wav", "mp3"], key="test")
    else:
        test_file = st.audio_input("Mikrofonla Kaydet")

    feat_test = None
    if test_file:
        st.audio(test_file)
        y_test = preprocess_audio_pipeline(test_file.getvalue())
        if y_test is not None:
            feat_test, viz_test = extract_features_mfcc(y_test)
            st.success("Test sesi işlendi.")

# DOĞRULAMA BUTONU
st.divider()
if st.button("🔍 KİMLİĞİ DOĞRULA", use_container_width=True, type="primary"):
    if model is None:
        st.error("Model yüklenemedi!")
    elif feat_ref is None or feat_test is None:
        st.warning("Lütfen her iki sesi de yükleyin.")
    else:
        with st.spinner("Siyam ağı karşılaştırıyor..."):
            # Tahmin
            distance = model.predict([feat_ref, feat_test], verbose=0)[0][0]
            
            st.metric("Hesaplanan Benzerlik Mesafesi", f"{distance:.4f}")
            
            if distance < THRESHOLD:
                st.success(f"✅ EŞLEŞME BAŞARILI! (Mesafe {THRESHOLD}'dan küçük)")
                st.balloons()
            else:
                st.error(f"⛔ EŞLEŞME BAŞARISIZ! (Mesafe {THRESHOLD}'dan büyük)")