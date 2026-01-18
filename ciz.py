import cfgrib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
import pandas as pd
import json
from scipy.ndimage import gaussian_filter

# --- AYARLAR ---
GRIB_FILE = 'fourcastnet.grib'

# DEĞİŞİKLİK BURADA: Klasör adını değiştirdik
OUTPUT_FOLDER = "ai_images" 
JSON_FILENAME = "ai_file_list.json"

# Klasör yoksa oluştur
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Minoconda library fix
os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('HOME')}/miniconda3/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

print(f"🤖 ECMWF AI (FourCastNet) VISUALIZER BAŞLATILIYOR...Hedef: {OUTPUT_FOLDER}/")

try:
    datasets = cfgrib.open_datasets(GRIB_FILE)
    
    ds_z = next((d for d in datasets if 'z' in d.data_vars), None)
    ds_msl = next((d for d in datasets if 'msl' in d.data_vars), None)
    ds_t = next((d for d in datasets if 't' in d.data_vars), None)
    
    if ds_z is None or ds_msl is None or ds_t is None:
        print("❌ Hata: GRIB dosyasında Z, MSL veya T parametreleri eksik!")
        exit()

    z_500 = ds_z['z'].sel(isobaricInhPa=500) / 98.0665
    msl = ds_msl['msl'] / 100.0
    t_850 = ds_t['t'].sel(isobaricInhPa=850) - 273.15

    steps_to_plot = range(0, z_500.step.size, 1)
    file_list_json = []

    print(f"✅ Toplam {len(steps_to_plot)} adım işlenecek.")

    for i in steps_to_plot:
        z_slice = z_500.isel(step=i)
        msl_slice = msl.isel(step=i)
        t_slice = t_850.isel(step=i)
        
        valid_time = pd.to_datetime(z_slice.valid_time.values)
        tarih_baslik = valid_time.strftime('%d.%m.%Y %H:%M')
        dosya_tarih = valid_time.strftime('%Y%m%d_%H%M00') 
        vade_saat = int(z_slice.step.values / np.timedelta64(1, 'h'))
        
        file_list_json.append(dosya_tarih)

        lons = z_slice.longitude.values
        lats = z_slice.latitude.values

        # 1. HARİTA: 500 hPa & MSLP
        fig = plt.figure(figsize=(18, 10), dpi=100)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([-80, 120, 20, 85], crs=ccrs.PlateCarree())

        z_smooth = gaussian_filter(z_slice.values, sigma=1.2)
        levels_z = np.arange(460, 604, 4) 
        cf = ax.contourf(lons, lats, z_smooth, levels=levels_z, cmap='turbo', extend='both')
        plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.03, shrink=0.6, aspect=40, label='500 hPa Geo. Yükseklik (dam)')

        msl_smooth = gaussian_filter(msl_slice.values, sigma=1.8)
        cs = ax.contour(lons, lats, msl_smooth, levels=np.arange(960, 1050, 4), colors='white', linewidths=1.0)
        plt.clabel(cs, inline=True, fmt='%d', fontsize=10, colors='white')

        plt.title(f"ECMWF AI | 500hPa & MSLP | {tarih_baslik} UTC (+{vade_saat}h)", fontsize=14, weight='bold', loc='left')
        plt.title("Kerem Palancı AI MODEL", fontsize=10, loc='right', color='black')
        ax.add_feature(cfeature.COASTLINE, edgecolor='black'); ax.add_feature(cfeature.BORDERS, alpha=0.5)

        fn_z = f"{OUTPUT_FOLDER}/GLOBAL_AIZ500_{dosya_tarih}.webp"
        plt.savefig(fn_z, dpi=100, bbox_inches='tight')
        plt.close()

        # 2. HARİTA: 850 hPa SICAKLIK
        fig = plt.figure(figsize=(18, 10), dpi=100)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent([-80, 120, 20, 85], crs=ccrs.PlateCarree())

        t_smooth = gaussian_filter(t_slice.values, sigma=1.0)
        levels_t = np.arange(-30, 40, 2)
        cf_t = ax.contourf(lons, lats, t_smooth, levels=levels_t, cmap='coolwarm', extend='both')
        plt.colorbar(cf_t, ax=ax, orientation='horizontal', pad=0.03, shrink=0.6, aspect=40, label='850 hPa Sıcaklık (°C)')

        plt.title(f"ECMWF AI | 850hPa Sıcaklık | {tarih_baslik} UTC (+{vade_saat}h)", fontsize=14, weight='bold', loc='left')
        plt.title("Kerem Palancı AI MODEL", fontsize=10, loc='right', color='black')
        ax.add_feature(cfeature.COASTLINE, edgecolor='black'); ax.add_feature(cfeature.BORDERS, alpha=0.5)

        fn_t = f"{OUTPUT_FOLDER}/GLOBAL_TEMP850_{dosya_tarih}.webp"
        plt.savefig(fn_t, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"🚀 Hazır (+{vade_saat}h) -> {OUTPUT_FOLDER}")

    # JSON OLUŞTURMA
    json_path = os.path.join(OUTPUT_FOLDER, JSON_FILENAME)
    with open(json_path, 'w') as f:
        json.dump(file_list_json, f)
    
    print(f"\n✅ İşlem Tamam! JSON ve resimler '{OUTPUT_FOLDER}' klasörüne atıldı.")

except Exception as e:
    print(f"❌ Kritik Hata: {e}")
