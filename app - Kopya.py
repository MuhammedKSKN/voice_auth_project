import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import noisereduce as nr
import io
import soundfile as sf
import matplotlib.pyplot as plt

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Sesle Kimlik Doğrulama & Deepfake Tespiti",
    page_icon="🎙️",
    layout="wide"
)

# --- BAŞLIK VE AÇIKLAMA ---
st.title("🛡️ Sesle Kimlik Doğrulama ve Deepfake Tespiti")
st.markdown("""
Bu uygulama, Siyam Evrişimli Sinir Ağı (Siamese CNN) kullanarak ses tabanlı kimlik doğrulama yapar.
Sistem, gerçek kullanıcı seslerini deepfake taklitlerinden ve yetkisiz kullanıcılardan ayırt etmek için tasarlanmıştır.
""")
st.markdown("---")

# --- KENAR ÇUBUĞU (SIDEBAR) AYARLARI ---
st.sidebar.header("⚙️ Ayarlar ve Model")

# 1. Eşik Değeri (Threshold) Ayarı
# Bu değerin altında kalan mesafeler "Eşleşme", üstünde kalanlar "Eşleşmeme" sayılır.
# Modelinizi test ederken bu değeri değiştirerek en iyi noktayı bulabilirsiniz.
THRESHOLD = st.sidebar.slider("Karar Eşik Değeri (Distance Threshold)", 0.0, 2.0, 0.5, 0.01)
st.sidebar.info(f"Mevcut Eşik: {THRESHOLD}. Bu değerin altı 'Doğrulandı' kabul edilir.")

# 2. Modeli Yükleme (Önbelleğe alma)
@st.cache_resource
def load_siamese_model():
    # MODEL YOLUNUZU BURAYA GİRİN
    model_path = 'model/best_siamese_model.h5' 
    try:
        model = tf.keras.models.load_model(model_path)
        st.sidebar.success("✅ Model başarıyla yüklendi!")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Model yüklenirken hata oluştu: {e}")
        return None

# Modeli başlat
model = load_siamese_model()

# --- YARDIMCI FONKSİYONLAR (METODOLOJİNİZE UYGUN) ---

def preprocess_audio_pipeline(audio_bytes, target_sr=16000, fixed_length_sec=4):
    """
    Metodolojide belirtilen 5 adımlı ön işleme hattı.
    Ham ses baytlarını alır, işlenmiş numpy dizisi döndürür.
    """
    try:
        # Byte verisini numpy dizisine çevir
        y, sr = sf.read(io.BytesIO(audio_bytes))
        
        # 1. Format Standardizasyonu (16kHz, Mono)
        if sr != target_sr:
            y = librosa.resample(y=y, orig_sr=sr, target_sr=target_sr)
        if len(y.shape) > 1:
             y = librosa.to_mono(y)
        
        # 2. Durağan Gürültü Azaltma (Stationary Noise Reduction)
        # Not: noisereduce bazen çok kısa seslerde sorun çıkarabilir, try-except eklenebilir.
        y = nr.reduce_noise(y=y, sr=target_sr)

        # 3. Sessizlik Silme (Silence Trimming)
        y, _ = librosa.effects.trim(y, top_db=20)

        # 4. Sabit Uzunluklu Segmentasyon (Fixed-Length Segmentation - 4sn)
        target_length = int(target_sr * fixed_length_sec) # 64000 örnek
        if len(y) < target_length:
            # Zero-padding (Kısa ise sıfır ekle)
            y = librosa.util.pad_center(y, size=target_length)
        else:
            # Truncation (Uzun ise kırp)
            y = y[:target_length]
            
        # 5. Normalizasyon (Genlik -1 ile 1 arası)
        y = librosa.util.normalize(y)

        return y, target_sr
    except Exception as e:
        st.error(f"Ses işleme hatası: {e}")
        return None, None

def extract_features_mfcc(processed_audio, sr=16000):
    """
    İşlenmiş sesten MFCC özelliklerini çıkarır ve model girişine uygun şekillendirir.
    Çıktı Boyutu: (1, 40, 128, 1)
    """
    # Metodolojideki parametreler
    n_mfcc = 40
    n_fft = 2048
    hop_length = 512
    
    mfcc = librosa.feature.mfcc(y=processed_audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    
    # MFCC genellikle (n_mfcc, zaman) şeklindedir.
    # Eğer 4 saniye ise ve hop_length 512 ise zaman boyutu yaklaşık 126-128 arası çıkar.
    # CNN girişi için sabit boyuta (örn: 128) emin olmak gerekebilir.
    # Burada librosa'nın çıktısının tam 128 zaman adımına denk geldiğini varsayıyoruz.
    # Değilse, burada da bir padding/trimming gerekebilir.
    
    # Şekillendirme: (Batch_Size, Height, Width, Channels) -> (1, 40, 128, 1)
    # Not: Zaman boyutunun 128 olduğundan emin olun, değilse eğitim kodunuza göre ayarlayın.
    if mfcc.shape[1] != 128:
        mfcc = librosa.util.fix_length(mfcc, size=128, axis=1)

    mfcc_reshaped = mfcc[np.newaxis, ..., np.newaxis]
    return mfcc_reshaped, mfcc # Görselleştirme için ham MFCC'yi de döndür

def calculate_euclidean_distance(embed1, embed2):
    """İki gömü vektörü arasındaki Öklid mesafesini hesaplar."""
    # Embeddings shape: (1, 128)
    return np.linalg.norm(embed1 - embed2)

def plot_spectrogram(mfcc_data, title):
    """MFCC görselleştirmesi için yardımcı fonksiyon."""
    fig, ax = plt.subplots(figsize=(4, 2))
    img = librosa.display.specshow(mfcc_data, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title=title)
    return fig

# --- ANA ARAYÜZ ---

col1, col2 = st.columns(2)

# --- SOL KOLON: REFERANS SES (ANCHOR) ---
with col1:
    st.header("1. Referans Ses (Anchor)")
    st.write("Yetkili kullanıcının gerçek sesi.")
    
    anchor_file = st.file_uploader("Referans ses dosyası yükle (.wav)", type=["wav", "mp3"], key="anchor")
    # Alternatif olarak mikrofondan da alınabilir ama anchor genellikle sabittir.
    
    anchor_processed = None
    anchor_features = None
    
    if anchor_file is not None:
        st.audio(anchor_file, format='audio/wav')
        with st.spinner('Referans ses işleniyor...'):
            # Byte verisini al
            anchor_bytes = anchor_file.getvalue()
            # Ön işleme
            anchor_processed, sr = preprocess_audio_pipeline(anchor_bytes)
            if anchor_processed is not None:
                # Öznitelik Çıkarımı
                anchor_features, anchor_mfcc_vis = extract_features_mfcc(anchor_processed, sr)
                st.success("Referans ses hazırlandı.")
                with st.expander("Spektrogramı Göster"):
                     st.pyplot(plot_spectrogram(anchor_mfcc_vis, "Anchor MFCC"))

# --- SAĞ KOLON: TEST SESİ ---
with col2:
    st.header("2. Test Sesi")
    st.write("Doğrulanacak şüpheli ses (Mikrofon veya Dosya).")

    # Yeni Streamlit özelliği: Ses Girişi (Mikrofon)
    test_audio_input = st.audio_input("Mikrofon ile Kaydet", key="test_mic")
    # Veya dosya yükleme
    test_file_upload = st.file_uploader("Veya test dosyası yükle", type=["wav", "mp3"], key="test_file")
    
    test_file = test_audio_input if test_audio_input else test_file_upload
    
    test_processed = None
    test_features = None

    if test_file is not None:
        st.audio(test_file, format='audio/wav')
        with st.spinner('Test sesi işleniyor...'):
             # Byte verisini al
            test_bytes = test_file.getvalue()
             # Ön işleme
            test_processed, sr = preprocess_audio_pipeline(test_bytes)
            if test_processed is not None:
                # Öznitelik Çıkarımı
                test_features, test_mfcc_vis = extract_features_mfcc(test_processed, sr)
                st.success("Test sesi hazırlandı.")
                with st.expander("Spektrogramı Göster"):
                     st.pyplot(plot_spectrogram(test_mfcc_vis, "Test MFCC"))

# --- DOĞRULAMA BÖLÜMÜ ---
st.markdown("---")
st.header("3. Doğrulama Sonucu")

verify_button = st.button("🔊 Kimliği Doğrula", type="primary", use_container_width=True)

if verify_button:
    if model is None:
        st.error("Model yüklenemediği için doğrulama yapılamıyor.")
    elif anchor_features is None or test_features is None:
        st.warning("Lütfen önce hem Referans hem de Test seslerini sağlayın.")
    else:
        with st.spinner('Siyam Ağı karşılaştırması yapılıyor...'):
            # NOT: Eğittiğiniz modelin çıktısına göre burası değişebilir.
            # SENARYO A: Modeliniz direkt mesafeyi (tek bir sayı) döndürüyorsa:
            # distance = model.predict([anchor_features, test_features])[0][0]
            
            # SENARYO B (Daha yaygın): Modeliniz iki ayrı embedding döndürüyorsa (Metodolojinize daha uygun):
            # Modelin iki çıktısı olduğunu varsayıyoruz: embedding_1, embedding_2
            embeddings = model.predict([anchor_features, test_features])
            embedding_anchor = embeddings[0]
            embedding_test = embeddings[1]
            
            # Öklid mesafesini hesapla
            distance = calculate_euclidean_distance(embedding_anchor, embedding_test)

            # --- SONUÇ EKRANI ---
            st.metric(label="Hesaplanan Benzerlik Mesafesi (Öklid)", value=f"{distance:.4f}")
            
            if distance < THRESHOLD:
                st.success("✅ KİMLİK DOĞRULANDI (Yetkili Kullanıcı)")
                st.balloons()
            else:
                st.error("⛔ KİMLİK REDDEDİLDİ (Potansiyel Sahtecilik/Deepfake)")