import os
import json

def generate_json():
    # Görsellerin olduğu klasör
    img_dir = "images"
    
    if not os.path.exists(img_dir):
        print(f"❌ Klasör bulunamadı: {img_dir}")
        return

    all_files = os.listdir(img_dir)
    time_stamps = set()

    for f in all_files:
        # Sadece webp ve png dosyalarını gör
        if (f.endswith(".webp") or f.endswith(".png")) and "_" in f:
            # Uzantıyı at ve parçalara ayır
            name_part = f.rsplit('.', 1)[0]
            parts = name_part.split("_")
            
            # Senin koddaki mantık: Son iki parça tarih_saat
            # Örn: TR_T2_20251231_120000 -> parts[-2]=20251231, parts[-1]=120000
            if len(parts) >= 2:
                date_p = parts[-2]
                time_p = parts[-1]
                
                # Tarih 8 hane, saat 6 hane olmalı (YYYYMMDD_HHMMSS)
                if len(date_p) == 8 and len(time_p) == 6:
                    ts = f"{date_p}_{time_p}"
                    time_stamps.add(ts)

    # Kronolojik sırala
    sorted_times = sorted(list(time_stamps))
    
    # JSON dosyasını kaydet
    with open(os.path.join(img_dir, "file_list.json"), "w") as jf:
        json.dump(sorted_times, jf, indent=2)
    
    print(f"✅ JSON Hazır: {len(sorted_times)} zaman dilimi eklendi.")

if __name__ == "__main__":
    generate_json()

