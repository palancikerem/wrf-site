#!/bin/bash

# 1. gh-pages dalına geç
git checkout gh-pages

# 2. GitHub'daki mevcut tüm resimleri 'git rm' ile siliyoruz 
# (Bu komut PC'ndeki dosyaları değil, Git'in takip ettiği kayıtları siler)
git rm -r images/*.png
git rm images/file_list.json

# 3. Şimdi kendi PC'ndeki güncel resimleri ve JSON'u tekrar ekle
git add images/*.png
git add images/file_list.json
git add index.html

# 4. Değişiklikleri kaydet ve GitHub'a 'temizlik yapıldı' diyerek gönder
git commit -m "Model Update: Eski veriler temizlendi, güncel haritalar yüklendi"
git push origin gh-pages --force

echo "İşlem tamam! GitHub temizlendi ve sadece yeni haritalar yüklendi."
