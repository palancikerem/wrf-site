import os
import json

def generate_json():
    img_dir = "images"
    all_files = os.listdir(img_dir)
    time_stamps = set()

    for f in all_files:
        if (f.endswith(".webp") or f.endswith(".png")) and "_" in f:
            parts = f.split(".")[0].split("_")
            # Dosya ismindeki tarih parçalarını yakala (Sondan önceki iki parça)
            # Örn: TR_T2_20251231_120000.webp -> 20251231 ve 120000
            if len(parts) >= 2:
                ts = f"{parts[-2]}_{parts[-1]}"
                if len(ts) >= 11: # YYYYMMDD_HH (en az)
                    time_stamps.add(ts)

    # ÖNEMLİ: Tüm yılları (2025 ve 2026) kronolojik sıraya sokar
    sorted_times = sorted(list(time_stamps))
    
    with open(os.path.join(img_dir, "file_list.json"), "w") as jf:
        json.dump(sorted_times, jf, indent=2)
    
    print(f"✅ JSON Hazır: {len(sorted_times)} zaman dilimi (2025 ve 2026 dahil) eklendi.")

if __name__ == "__main__":
    generate_json()
