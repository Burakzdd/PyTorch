#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 19:29:33 2021

@author: burakzdd
"""
"""
Bu çalışmada PyTorchVideo öğreticilerinden yararlanılmıştır.
https://pytorchvideo.org/docs/tutorial_torchhub_inference
"""
import torch
import json
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
from typing import Dict

# Kinetics 400 veri seti üzerinde eğitilmiş slowfast_r50 modeli ve modelin çalıştırılacağı cihaz seçelir.
device = "cpu"

model_name = "slowfast_r50"
model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)

# Model değerlendirilerek belirlenen cihaza (CPU) geçilir
model = model.to(device)
model = model.eval()

#dosya üzerinden etiket isimleri okunur.
with open("/home/burakzdd/Desktop/kinetics_classnames.json", "r") as f:
    kinetics_classnames = json.load(f)

# Etiket eşlemesi için bir kimlik numarası (id) oluşturulur
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")
    

####################
# SlowFast modelinin dönüşümü
####################

side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames = 32
sampling_rate = 2
frames_per_second = 30
alpha = 4

class PackPathway(torch.nn.Module):
    """
    Video kareleri bir tensör listesi olarak dönüştürülür
    """
    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Hızlı yoldan zamansal örnekleme gerçekleştirilir
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

transform =  ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x/255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(
                size=side_size
            ),            CenterCropVideo(crop_size),
            PackPathway()
        ]
    ),
)

# Giriş videosunun süresi de modele özeldir.
clip_duration = (num_frames * sampling_rate)/frames_per_second   

#Üzerinde çalışılacak olan video okunur
video_path = "/home/burakzdd/Downloads/pool.mp4"

#Belirtilen başlangıç ve bitiş süresi kullanılarak yüklenecek 
#videonun süresini belirlenir. start_sec, videoda eylemin 
#gerçekleştiği yere karşılık gelmektedir
start_sec = 0
end_sec = start_sec + clip_duration

# EncodingVideo yardımcı sınıfın başlatılır
video = EncodedVideo.from_path(video_path)

# İstenilen video yüklenir
video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

# Video girişini normalleştirmek için bir dönüşüm uygulanır
video_data = transform(video_data)

# Girişler çalışılmak istenilen cihaza taşınır
inputs = video_data["video"]
inputs = [i.to(device)[None, ...] for i in inputs] 

#Giriş videosu modelden geçirilir
preds = model(inputs)

#Tahminler alınır.
post_act = torch.nn.Softmax(dim=5)
preds = post_act(preds)
pred_classes = preds.topk(k=1).indices

# Tahmin edilen sınıflar etiket adlarıyla eşleştirilir
pred_class_names = [kinetics_id_to_classname[int(i)] for i in pred_classes[0]]
print("Tahmin edilen sınflar: %s" % ", ".join(pred_class_names))