# PyTorch
PyTorch, python tabanlı açık kaynaklı bir makine öğrenmesi kütüphanesidir. Grafik işlem birimlerinin (GPU) gücünü kullanarak daha güçlü ve daha hızlı tensör hesaplamaları ve sinir ağları oluşturmaktadır. Maksimum esneklik ve hız sağlama özelliğine sahip olduğu için geliştiricilerin işini oldukça kolaylaştıran bir kütüphanedir.

PyTorch özetle derin öğrenme ve yapay zeka uygulamalarının gerçekleştirilme şeklini değiştirebilme özelliğine sahip bir bilimsel işlev paketidir.

PyTorch’ tercih etme sebeplerinin başında;
  Grafik kullanımının daha hızlı olması
  Numpy kütüphanes ile olan uyumluluğu
  GPU’ lar için farklı back/end destekleri sunabilmesi
  Derlenmiş olan kodların kullanılarak C/C++’da da programlama yapabilme imkanı sağlaması
  Pythonic olması ve sinir ağ modellerini sorunsuz bir şekilde oluşturabilmesi
gibi sebepler yatmanktadır.

VERİLER ile GİRİŞ;
PyTorch’da veriler ile çalışırken etkinleştirmemiz gereken iki ana kütüphane vardır.
1. torch.utils.data.DataLoader
2. torch.utils.data.Dataset
Bunların haricinde TorchText , TorchVision ve TorchAudio gibi veri setlerini içerisnde bulunduran kütüphanelerde kullanılır. Bu çalışma da TorchVision kütüphanesi kullanılacaktır. Bu kütüphane gerçke dünyadaki görsel çoğu nesne sınıfı içerisinde barındırmaktadır.

MODEL OLUŞTURMA ve KAYDETME;
PyTorch’da bir model (sinir ağı) oluşturulurken nn.Module komutundan yararlanılır. Bunu bir örnek üzerinden açıklayalım;

#ilk olarak kütüphaneler aktif edilir
import torch
from torch import nn
Eğitimin GPU ya da CPU cihazlarından hangisi üzerinden yapıldığına bakılır.
#Eğitim için GPU’da mı yoksa CPU’da mı çalışıldığı kontrol edilir.
device = “cuda” if torch.cuda.is_available() else “cpu”
print(“{} cihazı kullanılıyor”.format(device))

__init__ fonksiyonu ile fonksiyonda ağın katmanları tanımlanır. forward fonksiyonuyla ise verilerin ağ üzerinden nasıl geçeceği belirlenir.
# Model tanımlanır
class YapaySinirAgi(nn.Module):
      def __init__(self):
          super(YapaySinirAgi, self).__init__()
          self.flatten = nn.Flatten()
          self.linear_relu_stack = nn.Sequential(
              nn.Linear(28*28, 512),
              nn.ReLU(),
              nn.Linear(512, 512),
              nn.ReLU(),
              nn.Linear(512, 10),
              nn.ReLU()
           )
      def forward(self, x):
          x = self.flatten(x)
          logits = self.linear_relu_stack(x)
          return logits
          
Son olarakta eğer varsa oluşturulan modeli GPU’ya alıp bastırıyoruz.
model = YapaySinirAgi().to(device)
print(model)

Bu oluşturduğumuz modeli kaydetmek istersek torch kütüphanesi içerisindeki save parametresini kullanırız.
torch.save(model.state_dict(), “model.pth”)
print(“PyTorch modeli model.pth olarak kaydedildi”)

