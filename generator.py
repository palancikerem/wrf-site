import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import glob, os, json, datetime, multiprocessing, warnings, gc
from netCDF4 import Dataset
from wrf import (getvar, to_np, latlon_coords, cartopy_xlim, cartopy_ylim, 
                 get_cartopy, smooth2d, interplevel)
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import numpy as np
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ AYARLAR
# ==========================================
BASE_DIR = "/home/kerem/WRF_RUN_20251225_12Z"
OUTPUT_DIR = "/home/kerem/wrf_site/images"

REGIONS = {
    "TR": None,
    "MARMARA": [25.5, 32.5, 39.5, 43.5],
    "EGE": [24.0, 31.0, 35.5, 40.5],
    "AKDENIZ": [27.0, 37.0, 34.0, 38.5],
    "KARADENIZ": [32.0, 42.5, 40.0, 44.5],
    "DOGU_ANADOLU": [36.5, 45.0, 36.0, 42.5],
    "IC_ANADOLU": [29.0, 37.0, 37.0, 41.0]
}

# ==========================================
# 🎨 RENK PALETLERİ & HASSAS SKALALAR
# ==========================================
def create_wind_cmap():
    colors = ['#A020F0', '#7B00FF', '#0000FF', '#007FFF', '#00FFFF', '#00FF7F', '#00FF00', 
              '#7FFF00', '#FFFF00', '#FFD700', '#FFA500', '#FF7F00', '#FF4500', '#FF0000', 
              '#D20000', '#A50000', '#7B0000', '#B03060', '#FF1493', '#FF69B4']
    return mcolors.LinearSegmentedColormap.from_list("custom_wind", colors)

def create_temperature_cmap():
    neg_colors = ['#2d004b', '#4d004b', '#542788', '#8073ac', '#b2abd2', '#d8daeb', '#08306b', '#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#f7fbff']
    pos_colors = ['#006400', '#008000', '#228b22', '#32cd32', '#7cfc00', '#adff2f', '#dfff00', '#ffff00', '#ffd700', '#ffcc00', '#ffb900', '#ffa500', '#ff8c00', '#ff7f00', '#ff4500', '#ff0000', '#e60000', '#cc0000', '#990000', '#660000']
    return mcolors.ListedColormap(neg_colors + pos_colors)

def create_rain_cmap():
    colors_hex = ['#dedef2', '#b4d7ff', '#75baff', '#359aff', '#0482ff', '#0069d2', '#00367f', '#148f1b', '#1acf05', '#63ed07', '#fff42b', '#e8dc00', '#f06000', '#ff7f27', '#ffa66a', '#f84e78', '#f71e54', '#db0f2a', '#a30000', '#880000', '#64007f', '#c200fb', '#dd66ff', '#eba6ff', '#f9e6ff']
    return mcolors.ListedColormap(colors_hex)

def get_pro_settings(var_code):
    settings = {"extend": "max", "norm": None, "contour_levels": None}
    
    # --- SICAKLIKLAR ---
    if var_code in ["T2", "TEMP_850", "TEMP_500", "HEAT_INDEX"]:
        settings.update({"levels": np.arange(-30, 43, 1), "cmap": create_temperature_cmap(), "unit": "°C", "extend": "both"})
        if var_code in ["TEMP_850", "TEMP_500"]: settings["contour_levels"] = np.arange(-40, 45, 2)
        titles = {"T2":"2m Sıcaklık","TEMP_850":"850hPa Sıcaklık","TEMP_500":"500hPa Sıcaklık","HEAT_INDEX":"Hissedilen Sıcaklık"}
        settings["title_tr"] = titles.get(var_code)

    # --- YAĞIŞ & KAR (ULTRA HASSAS SKALA) ---
    elif var_code in ["RAIN", "RAIN1H", "SNOW", "SNOWDEPTH"]:
        settings["cmap"] = create_rain_cmap()
        if var_code == "RAIN": 
            settings["levels"] = [0.1, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200, 250, 300, 400, 500]
            settings["title_tr"] = "Toplam Yağış"
        elif var_code == "RAIN1H": 
            settings["levels"] = [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60]
            settings["title_tr"] = "Saatlik Yağış"
        elif var_code == "SNOW": 
            settings["levels"] = [0.1, 0.5, 1, 2, 3, 5, 7, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 100, 125, 150]
            settings["title_tr"] = "Toplam Kar (cm)"
        elif var_code == "SNOWDEPTH": 
            settings["levels"] = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 150]
            settings["title_tr"] = "Kar Derinliği (cm)"
        settings["norm"] = mcolors.BoundaryNorm(settings["levels"], settings["cmap"].N)

    # --- RÜZGAR & DİĞERLERİ ---
    elif var_code == "UVMET10": 
        settings.update({"levels": [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 100, 120, 150], "cmap": create_wind_cmap(), "title_tr": "10m Hamle", "unit": "km/h"})
    elif var_code == "CAPE": 
        settings.update({"levels": [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000], "cmap": plt.cm.YlOrRd, "title_tr": "CAPE", "unit": "J/kg"})
    elif var_code == "MDBZ": 
        settings.update({"levels": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], "cmap": plt.cm.jet, "title_tr": "Radar", "unit": "dBZ"})
    elif var_code == "SLP": 
        settings.update({"levels": np.arange(980, 1041, 2), "cmap": plt.cm.RdYlBu_r, "title_tr": "Basınç", "unit": "hPa", "contour_levels": np.arange(960, 1050, 4), "extend": "both"})
    
    return settings

def process_single_file(file_info):
    f, idx, files, dom = file_info
    if not os.path.exists(f) or os.path.getsize(f) == 0: return None
    try:
        nc = Dataset(f, "r")
        dt = datetime.datetime.strptime(os.path.basename(f).replace(f"wrfout_{dom}_", ""), "%Y-%m-%d_%H:%M:%S")
        clean_time = dt.strftime("%Y%m%d_%H%M%S")
        print(f"-> {dt.strftime('%d.%m %H:%M')}Z İşleniyor...")

        p_ref = getvar(nc, "T2")
        lats, lons = latlon_coords(p_ref)
        cart_proj = get_cartopy(p_ref)
        
        var_list = ["T2", "TEMP_850", "TEMP_500", "HEAT_INDEX", "RAIN", "RAIN1H", "SNOW", "SNOWDEPTH", "UVMET10", "CAPE", "MDBZ", "SLP"]

        try:
            shapefile = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_1_states_provinces')
            provinces = list(shpreader.Reader(shapefile).records())
        except: provinces = []

        for var_code in var_list:
            try:
                settings = get_pro_settings(var_code)
                var_data, u, v = None, None, None
                
                # Değişken Çekme
                if var_code == "T2": var_data = getvar(nc, "T2") - 273.15
                elif var_code == "TEMP_850": var_data = interplevel(getvar(nc, "tc"), getvar(nc, "pressure"), 850.0)
                elif var_code == "TEMP_500": var_data = interplevel(getvar(nc, "tc"), getvar(nc, "pressure"), 500.0)
                elif var_code == "HEAT_INDEX":
                    t_c = getvar(nc, "T2") - 273.15
                    rh = getvar(nc, "rh2")
                    var_data = t_c + 0.5555 * (6.11 * np.exp(5417.7530 * (1/273.16 - 1/(t_c + 273.15))) * (rh/100) - 10)
                elif var_code == "RAIN": var_data = getvar(nc, "RAINC") + getvar(nc, "RAINNC")
                elif var_code == "RAIN1H":
                    curr = getvar(nc, "RAINC") + getvar(nc, "RAINNC")
                    if idx > 0:
                        with Dataset(files[idx-1], "r") as pnc: var_data = curr - (getvar(pnc, "RAINC") + getvar(pnc, "RAINNC"))
                    else: var_data = curr
                elif var_code == "SNOW": var_data = getvar(nc, "SNOWNC")
                elif var_code == "SNOWDEPTH": var_data = getvar(nc, "SNOWH") * 100.0
                elif var_code == "UVMET10": u, v = getvar(nc, "uvmet10"); var_data = np.sqrt(u**2 + v**2) * 3.6
                elif var_code == "CAPE": var_data = getvar(nc, "cape_2d")[0]
                elif var_code == "MDBZ": var_data = getvar(nc, "mdbz")
                elif var_code == "SLP": var_data = getvar(nc, "slp")

                if var_data is None: continue
                
                # 0 Maskeleme (BEYAZ BIRAKMA)
                if var_code not in ["T2", "TEMP_850", "TEMP_500", "SLP", "HEAT_INDEX"]:
                    var_data = np.where(var_data < settings["levels"][0], np.nan, var_data)

                for reg_name, extent in REGIONS.items():
                    save_path = os.path.join(OUTPUT_DIR, f"{reg_name}_{var_code}_{clean_time}.png")
                    fig = plt.figure(figsize=(12, 10)); ax = plt.axes(projection=cart_proj)
                    if extent: ax.set_extent(extent, crs=ccrs.PlateCarree())
                    
                    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, zorder=3); ax.add_feature(cfeature.BORDERS, linewidth=0.8, zorder=3)
                    for prov in provinces:
                        if prov.attributes.get('admin') == 'Turkey':
                            ax.add_geometries([prov.geometry], ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=0.4, alpha=0.3, zorder=3)

                    cf = ax.contourf(to_np(lons), to_np(lats), to_np(var_data), levels=settings["levels"], 
                                   norm=settings.get("norm"), cmap=settings["cmap"], transform=ccrs.PlateCarree(), extend=settings["extend"])
                    
                    if u is not None:
                        ax.streamplot(to_np(lons), to_np(lats), to_np(u), to_np(v), transform=ccrs.PlateCarree(), color='white', linewidth=0.6, density=2.0, zorder=4)

                    if settings.get("contour_levels") is not None:
                        cs = ax.contour(to_np(lons), to_np(lats), smooth2d(to_np(var_data), 3), levels=settings["contour_levels"], colors="black", linewidths=0.6, transform=ccrs.PlateCarree())
                        ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')

                    plt.colorbar(cf, ax=ax, shrink=0.75, pad=0.02)
                    
                    # BAŞLIKLAR (GFS 0.25 -> 6km UNUTULMADI)
                    ax.set_title(f"{reg_name} - {settings['title_tr']}", loc='left', fontweight='bold')
                    ax.set_title("GFS 0.25° ➔ 6km", loc='center', fontweight='bold', color='blue')
                    ax.set_title(f"{dt.strftime('%d.%m %H:%M')}Z", loc='right', color='red', fontweight='bold')
                    ax.text(0.01, -0.04, "Kerem Palancı | WRF Model", transform=ax.transAxes, fontsize=10, fontweight='bold')
                    
                    plt.savefig(save_path, dpi=110, bbox_inches='tight'); plt.close(fig)
            except Exception as e: print(f"Hata {var_code}: {e}"); plt.close('all')
        nc.close()
    except Exception as e: print(f"Dosya Hatası: {e}")
    gc.collect()
    return clean_time

def generate_maps():
    print(" WRF Engine")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(BASE_DIR, "wrfout_d01_*")))
    file_tasks = [(f, idx, files, "d01") for idx, f in enumerate(files)]
    with multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count()-1)) as pool:
        results = pool.map(process_single_file, file_tasks)
    with open(os.path.join(OUTPUT_DIR, "file_list.json"), "w") as jf:
        json.dump(sorted(list(set(results))), jf)

if __name__ == "__main__":
    generate_maps()