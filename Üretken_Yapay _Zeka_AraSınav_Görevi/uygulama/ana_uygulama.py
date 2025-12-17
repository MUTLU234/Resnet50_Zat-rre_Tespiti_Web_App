import streamlit as st
from PIL import Image
# Aynı klasördeki model_islemleri.py dosyasından ZaturreModeli sınıfını çağırıyoruz
from model_islemleri import ZaturreModeli

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Zatürre Tespiti Asistanı",
    page_icon="🫁",
    layout="centered"
)

# Başlık ve Açıklama
st.title("🫁 Yapay Zeka Destekli Zatürre Tespiti")
st.markdown("""
Bu uygulama, derin öğrenme (ResNet50) kullanarak göğüs röntgeni görüntülerinde zatürre belirtilerini analiz eder.
**Lütfen bir X-Ray görüntüsü yükleyin.**
""")

# Modeli Önbellekleme (Cache) - Her seferinde yeniden yüklenmesin diye
@st.cache_resource
def modeli_getir():
    return ZaturreModeli()

# Modeli Yükle
try:
    yapay_zeka = modeli_getir()
    st.success("Yapay Zeka Modeli Hazır!")
except Exception as e:
    st.error(f"Model yüklenirken bir sorun oluştu: {e}")
    st.stop()

# Dosya Yükleme Alanı
yuklenen_dosya = st.file_uploader("Bir Göğüs Röntgeni (X-Ray) Seçin", type=["jpg", "jpeg", "png"])

if yuklenen_dosya is not None:
    # Resmi Göster
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(yuklenen_dosya, caption='Yüklenen Görüntü', use_container_width=True)

    with col2:
        st.write("Analiz ediliyor...")
        # Tahmin Yap
        bar = st.progress(0)
        tahmin, guven = yapay_zeka.tahmin_et(yuklenen_dosya)
        bar.progress(100)

        # Sonucu Göster
        if tahmin == "Zatürre":
            st.error(f"**Sonuç:** {tahmin}")
            st.warning(f"**Güven Skoru:** %{guven:.2f}")
            st.markdown("⚠️ *Lütfen en kısa sürede bir doktora başvurun.*")
        elif tahmin == "Normal":
            st.success(f"**Sonuç:** {tahmin}")
            st.info(f"**Güven Skoru:** %{guven:.2f}")
            st.markdown("✅ *Herhangi bir bulguya rastlanmadı.*")
        else:
            st.error("Görüntü analiz edilemedi.")

        # --- YENİ ÖZELLİK: Grad-CAM (Açıklanabilir Yapay Zeka) ---
        st.markdown("---")
        st.subheader("🔍 Detaylı Analiz (Doktor Modu)")
        if st.checkbox("Yapay Zekanın Nereye Baktığını Göster (Isı Haritası)"):
            with st.spinner("Isı haritası oluşturuluyor..."):
                isi_haritasi = yapay_zeka.isi_haritasi_olustur(yuklenen_dosya)
                
                if isi_haritasi is not None:
                    st.image(isi_haritasi, caption="Modelin Odaklandığı Bölgeler (Kırmızı Alanlar)", use_container_width=True)
                    st.info("ℹ️ Kırmızı alanlar, modelin 'Zatürre' veya 'Normal' kararı verirken en çok dikkate aldığı bölgelerdir.")
                else:
                    st.warning("Isı haritası oluşturulamadı. Gerekli kütüphaneler eksik olabilir.")

    # Detaylı Bilgi (İsteğe bağlı)
    st.markdown("---")
    st.caption("Not: Bu sistem sadece bir yardımcı araçtır ve kesin tıbbi teşhis koyamaz. Sonuçlar eğitim verisine bağlı olarak değişiklik gösterebilir.")
