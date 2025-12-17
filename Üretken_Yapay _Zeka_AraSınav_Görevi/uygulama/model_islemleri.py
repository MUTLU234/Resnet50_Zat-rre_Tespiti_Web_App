import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

class ZaturreModeli:
    def __init__(self):
        self.model = None
        self.cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.siniflar = ["Normal", "Zatürre"]
        self._modeli_yukle()

    def _modeli_yukle(self):
        """
        Ön eğitimli ResNet50 modelini yükler.
        Eğer 'zaturre_modeli.pth' dosyası varsa eğitilmiş ağırlıkları yükler.
        Yoksa ImageNet ağırlıklarıyla devam eder (Eğitimsiz mod).
        """
        try:
            # Temel Mimariyi Kur
            agirliklar = models.ResNet50_Weights.DEFAULT
            self.model = models.resnet50(weights=agirliklar)
            
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, 2)
            
            # Eğitilmiş Model Kontrolü
            import os
            model_yolu = "uygulama/zaturre_modeli.pth"
            if os.path.exists(model_yolu):
                try:
                    state_dict = torch.load(model_yolu, map_location=self.cihaz)
                    self.model.load_state_dict(state_dict)
                    print(f"✅ Eğitilmiş model yüklendi: {model_yolu}")
                except Exception as e:
                    print(f"⚠️ Model dosyası bulundu ama yüklenemedi: {e}")
                    print("Varsayılan (ImageNet) ağırlıklar kullanılıyor.")
            else:
                print("ℹ️ Eğitilmiş model bulunamadı. Varsayılan (ImageNet) ağırlıklar kullanılıyor.")

            self.model = self.model.to(self.cihaz)
            self.model.eval()
            print("Model başarıyla hazırlandı.")
        except Exception as e:
            print(f"Model yüklenirken kritik hata: {e}")

    def tahmin_et(self, resim_dosyasi):
        """
        Verilen resim dosyasını (PIL Image) alır, işler ve tahmin sonucunu döndürür.
        """
        if self.model is None:
            return "Model Yüklenemedi", 0.0

        try:
            # Resim Ön İşleme (Preprocessing)
            islem = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            resim = Image.open(resim_dosyasi).convert('RGB')
            resim_tensor = islem(resim).unsqueeze(0) # Batch boyutu ekle (1, 3, 224, 224)
            resim_tensor = resim_tensor.to(self.cihaz)

            with torch.no_grad():
                ciktilar = self.model(resim_tensor)
                # Softmax ile olasılık değerlerine çevir
                olasiliklar = torch.nn.functional.softmax(ciktilar, dim=1)
                
                # En yüksek olasılığa sahip sınıfı ve skoru al
                en_yuksek_skor, tahmin_indeksi = torch.max(olasiliklar, 1)
                
                tahmin_sinifi = self.siniflar[tahmin_indeksi.item()]
                skor = en_yuksek_skor.item() * 100 # Yüzdeye çevir

            return tahmin_sinifi, skor

        except Exception as e:
            print(f"Tahmin sırasında hata: {e}")
    
    def isi_haritasi_olustur(self, resim_dosyasi):
        """
        Grad-CAM tekniği ile modelin odaklandığı bölgeleri gösteren bir ısı haritası oluşturur.
        """
        try:
            from pytorch_grad_cam import GradCAM
            from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
            from pytorch_grad_cam.utils.image import show_cam_on_image
            import numpy as np
            import cv2
        except ImportError:
            print("Grad-CAM kütüphanesi eksik!")
            return None

        if self.model is None:
            return None

        # Hedef Katman: ResNet'in son konvolüsyonel bloğu (layer4[-1])
        hedef_katmanlar = [self.model.layer4[-1]]
        
        # Resmi Hazırla
        islem = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        orijinal_resim = Image.open(resim_dosyasi).convert('RGB')
        resim_tensor = islem(orijinal_resim).unsqueeze(0).to(self.cihaz)

        # Grad-CAM Nesnesi
        cam = GradCAM(model=self.model, target_layers=hedef_katmanlar)

        # Hangi sınıf için bakacağız? (En yüksek ihtimalli sınıf için)
        ciktilar = self.model(resim_tensor)
        tahmin_indeksi = torch.argmax(ciktilar, dim=1).item()
        hedefler = [ClassifierOutputTarget(tahmin_indeksi)]

        # Haritayı Oluştur
        grayscale_cam = cam(input_tensor=resim_tensor, targets=hedefler)
        grayscale_cam = grayscale_cam[0, :] # İlk batch'i al

        # Görüntüleme için orijinal resmi numpy (0-1 arası) formatına çevir
        resim_np = np.array(orijinal_resim.resize((224, 224)))
        resim_np = resim_np / 255.0
        
        # Isı haritasını orjinal resim üzerine bindir (Superimpose)
        visualization = show_cam_on_image(resim_np, grayscale_cam, use_rgb=True)
        
        return visualization
