import matplotlib

matplotlib.use('Agg') 

import matplotlib.pyplot as plt

import glob

import os

import json

import datetime

from netCDF4 import Dataset

from wrf import (getvar, to_np, latlon_coords, cartopy_xlim, cartopy_ylim, 

                 get_cartopy, smooth2d, interplevel, destagger, cape_2d)

import cartopy.crs as ccrs

import cartopy.feature as cfeature

import numpy as np

import matplotlib.colors as mcolors

import warnings

import gc



warnings.filterwarnings("ignore")



# ==========================================

# ⚙️ AYARLAR

# ==========================================

BASE_DIR = "/home/kerem/WRF_Build/WRFV4.5.2/test/em_real"

OUTPUT_DIR = "/home/kerem/wrf_site/images"

DOMAINS = ["d01", "d02", "d03"]

DOMAIN_RES = {"d01": 9, "d02": 3, "d03": 1}

ONLY_3H = False

MS_TO_KT = 1.94384



# ==========================================

# 🎨 RENK PALETLERİ VE AYARLAR

# ==========================================



def create_temperature_cmap():

    colors_hex = ['#4B0082', '#6A0DAD', '#8B00FF', '#9932CC', '#0000CD', '#0000FF', '#1E90FF', '#4169E1', '#00BFFF', '#00CED1', '#00E5EE', '#00FFFF', '#7FFFD4', '#98FB98', '#90EE90', '#00FF00', '#ADFF2F', '#FFFF00', '#FFD700', '#FFA500', '#FF8C00', '#FF6347', '#FF4500', '#FF0000', '#DC143C', '#B22222', '#8B0000', '#800000', '#654321', '#4B3621', '#3D2817']

    return mcolors.LinearSegmentedColormap.from_list("temp_pro", colors_hex, N=256)



def create_rain_cmap(n_levels):

    colors = ['#F0F8FF', '#D6EAF8', '#AED6F1', '#85C1E2', '#5DADE2', '#3498DB', '#2E86C1', '#1F618D', '#52BE80', '#27AE60', '#F4D03F', '#F39C12', '#E67E22', '#D35400', '#C0392B', '#922B21', '#8E44AD', '#4A235A']

    return colors[:n_levels]



def create_wind_cmap():

    return ['#FFFFFF', '#F0F8FF', '#E1F5FE', '#B3E5FC', '#81D4FA', '#4FC3F7', '#29B6F6', '#03A9F4', '#66BB6A', '#9CCC65', '#FFEB3B', '#FFC107', '#FF9800', '#FF5722', '#F44336', '#E91E63', '#9C27B0', '#673AB7']



def create_cape_colors():

    return ['#E8F5E9', '#C8E6C9', '#81C784', '#4CAF50', '#FFF176', '#FFB74D', '#FF7043', '#E53935', '#B71C1C', '#8E44AD', '#D500F9', '#FF4081']



def create_snow_colors():

    return ['#E3F2FD', '#BBDEFB', '#64B5F6', '#2196F3', '#1976D2', '#0D47A1', '#5E35B1', '#7B1FA2', '#AD1457', '#C2185B', '#FF4081']



def create_cloud_cmap():

    colors_hex = ['#FFFFFF', '#F5F5F5', '#E0E0E0', '#BDBDBD', '#9E9E9E', '#757575', '#616161', '#424242', '#303030']

    return mcolors.LinearSegmentedColormap.from_list("cloud", colors_hex, N=256)



def get_pro_settings(var_code, data_np, domain):

    settings = {}

    res_km = DOMAIN_RES[domain]

    

    # --- YARDIMCI: Güvenli Min/Max Hesaplayıcı ---

    def get_safe_limits(data, percentile_min=1, percentile_max=99, default_min=0, default_max=10):

        try:

            vmin = np.nanpercentile(data, percentile_min)

            vmax = np.nanpercentile(data, percentile_max)

            if np.isnan(vmin) or np.isnan(vmax):

                return default_min, default_max

            if vmin == vmax:

                return vmin, vmin + 1

            return vmin, vmax

        except:

            return default_min, default_max



    if var_code == "T2":

        settings["cmap"] = create_temperature_cmap()

        settings["title_tr"] = "2m Sıcaklık"

        settings["unit"] = "°C"

        vmin, vmax = get_safe_limits(data_np, default_min=-5, default_max=30)

        settings["vmin"], settings["vmax"] = max(-30, np.floor(vmin/5)*5), min(45, np.ceil(vmax/5)*5)

        settings["levels"] = np.linspace(settings["vmin"], settings["vmax"], 100)

        settings["contour_levels"] = np.arange(np.floor(vmin), np.ceil(vmax)+1, 2)

        settings["contour_colors"], settings["contour_linewidths"], settings["contour_alpha"] = 'black', 0.4, 0.4

        settings["extend"], settings["norm"] = "both", None



    elif var_code == "RAIN":

        settings["title_tr"], settings["unit"] = "Toplam Yağış", "mm"

        levels = [0.1, 0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 60, 80, 100, 125, 150]

        colors = create_rain_cmap(len(levels)-1)

        settings["cmap"] = mcolors.ListedColormap(colors)

        settings["cmap"].set_under('white', alpha=0); settings["cmap"].set_over('#2C1A40')

        settings["levels"], settings["norm"], settings["extend"] = levels, mcolors.BoundaryNorm(levels, len(colors)), "max"

        settings["contour_levels"] = None



    elif var_code == "UVMET10":

        settings["title_tr"], settings["unit"] = "10m Rüzgar", "knot"

        levels = [3, 5, 8, 12, 16, 20, 24, 28, 32, 36, 40, 45, 50, 55, 60, 70, 80, 100]

        colors = create_wind_cmap()

        settings["cmap"] = mcolors.ListedColormap(colors[:len(levels)-1])

        settings["cmap"].set_under('white', alpha=0); settings["cmap"].set_over('#311B92')

        settings["levels"], settings["norm"], settings["extend"] = levels, mcolors.BoundaryNorm(levels, len(colors)), "max"

        settings["contour_levels"] = None



    elif var_code == "CAPE":

        settings["title_tr"], settings["unit"] = "CAPE (MUCAPE)", "J/kg"

        levels = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000]

        colors = create_cape_colors()

        settings["cmap"] = mcolors.ListedColormap(colors)

        settings["cmap"].set_under('white', alpha=0); settings["cmap"].set_over('#FFFFFF')

        settings["levels"], settings["norm"], settings["extend"] = levels, mcolors.BoundaryNorm(levels, len(colors)), "max"

        settings["contour_levels"] = None



    elif var_code == "CIN":

        settings["title_tr"], settings["unit"] = "CIN", "J/kg"

        settings["cmap"] = plt.cm.YlOrRd_r

        settings["vmin"], settings["vmax"] = -500, 0

        settings["levels"], settings["contour_levels"], settings["extend"], settings["norm"] = np.linspace(-500, 0, 50), None, "min", None



    elif var_code == "SRH":

        settings["title_tr"], settings["unit"] = "0-3km SRH", "m²/s²"

        levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000]

        colors = ['#FFFFCC', '#FFEDA0', '#FED976', '#FEB24C', '#FD8D3C', '#FC4E2A', '#E31A1C', '#BD0026', '#800026', '#4D0013']

        settings["cmap"] = mcolors.ListedColormap(colors)

        settings["cmap"].set_under('white', alpha=0); settings["cmap"].set_over('#1A0008')

        settings["levels"], settings["norm"], settings["extend"] = levels, mcolors.BoundaryNorm(levels, len(colors)), "max"

        settings["contour_levels"] = None



    elif var_code == "SHEAR":

        settings["title_tr"], settings["unit"] = "0-6km Bulk Shear", "kt"

        levels = [20, 30, 40, 50, 60, 70, 80, 90, 100]

        colors = ['#E0F7FA', '#B2EBF2', '#4DD0E1', '#00BCD4', '#0097A7', '#006064', '#FF9800', '#E65100', '#BF360C']

        settings["cmap"] = mcolors.ListedColormap(colors)

        settings["cmap"].set_under('white', alpha=0); settings["cmap"].set_over('#3E2723')

        settings["levels"], settings["norm"], settings["extend"] = levels, mcolors.BoundaryNorm(levels, len(colors)), "max"

        settings["contour_levels"] = [40, 60]; settings["contour_colors"] = 'black'; settings["contour_linewidths"] = 0.5; settings["contour_alpha"] = 0.6



    elif var_code == "CLOUD":

        settings["title_tr"], settings["unit"] = "Toplam Bulutluluk", "%"

        settings["cmap"] = create_cloud_cmap()

        settings["vmin"], settings["vmax"] = 0, 100

        settings["levels"] = np.linspace(0, 100, 21)

        settings["contour_levels"], settings["extend"], settings["norm"] = None, "neither", None



    elif var_code == "TD2":

        settings["cmap"], settings["title_tr"], settings["unit"] = plt.cm.BrBG, "2m Çiy Noktası", "°C"

        vmin, vmax = get_safe_limits(data_np, default_min=-5, default_max=20)

        settings["vmin"], settings["vmax"] = np.floor(vmin/5)*5, np.ceil(vmax/5)*5

        settings["levels"] = np.linspace(settings["vmin"], settings["vmax"], 100)

        settings["contour_levels"] = np.arange(settings["vmin"], settings["vmax"]+1, 2)

        settings["contour_colors"], settings["contour_linewidths"], settings["contour_alpha"] = 'black', 0.4, 0.4

        settings["extend"], settings["norm"] = "both", None



    elif var_code == "HEAT_INDEX":

        settings["cmap"], settings["title_tr"], settings["unit"] = create_temperature_cmap(), "Hissedilen Sıcaklık", "°C"

        vmin, vmax = get_safe_limits(data_np, default_min=0, default_max=35)

        settings["vmin"], settings["vmax"] = max(-10, np.floor(vmin/5)*5), min(60, np.ceil(vmax/5)*5)

        settings["levels"] = np.linspace(settings["vmin"], settings["vmax"], 100)

        settings["contour_levels"] = np.arange(settings["vmin"], settings["vmax"]+1, 2)

        settings["contour_colors"], settings["contour_linewidths"], settings["contour_alpha"] = 'black', 0.4, 0.4

        settings["extend"], settings["norm"] = "both", None



    elif var_code == "RH2":

        settings["title_tr"], settings["unit"] = "2m Bağıl Nem", "%"

        settings["cmap"] = plt.cm.BrBG

        settings["vmin"], settings["vmax"] = 0, 100

        settings["levels"] = np.linspace(0, 100, 50)

        settings["contour_levels"] = [20, 40, 60, 80]

        settings["contour_colors"], settings["contour_linewidths"], settings["contour_alpha"], settings["extend"], settings["norm"] = 'black', 0.4, 0.4, "neither", None



    elif var_code == "SNOWH":

        settings["title_tr"], settings["unit"] = "Toplam Kar Yağışı", "mm"

        levels = [0.5, 1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60]

        colors = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6', '#2196F3', '#1976D2', '#1565C0', '#5E35B1', '#7B1FA2', '#AD1457', '#FF00FF']

        settings["cmap"] = mcolors.ListedColormap(colors)

        settings["cmap"].set_under('white', alpha=0); settings["cmap"].set_over('#FFFFFF')

        settings["levels"], settings["norm"], settings["extend"] = levels, mcolors.BoundaryNorm(levels, len(colors)), "max"

        settings["contour_levels"] = None



    elif var_code == "SNOWDEPTH":

        settings["title_tr"], settings["unit"] = "Kar Derinliği", "cm"

        levels = [1, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]

        colors = create_snow_colors()

        settings["cmap"] = mcolors.ListedColormap(colors)

        settings["cmap"].set_under('white', alpha=0); settings["cmap"].set_over('#FFFFFF')

        settings["levels"], settings["norm"], settings["extend"] = levels, mcolors.BoundaryNorm(levels, len(colors)), "max"

        settings["contour_levels"] = None



    elif var_code == "MDBZ":

        settings["title_tr"], settings["unit"] = "Kompozit Radar", "dBZ"

        levels = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

        colors = ['#646464', '#04E9E7', '#019FF4', '#0300F4', '#02FD02', '#01C501', '#008E00', '#FDF802', '#E5BC00', '#FD9500', '#FD0000', '#D40000', '#BC0000', '#E700FF']

        settings["cmap"] = mcolors.ListedColormap(colors)

        settings["cmap"].set_under('white', alpha=0); settings["cmap"].set_over('#FFFFFF')

        settings["levels"], settings["norm"], settings["extend"] = levels, mcolors.BoundaryNorm(levels, len(colors)), "max"

        settings["contour_levels"] = None



    elif var_code == "SLP":

        settings["title_tr"], settings["unit"] = "Deniz Seviyesi Basıncı", "hPa"

        settings["cmap"] = plt.cm.RdYlBu_r

        vmin, vmax = get_safe_limits(data_np, default_min=990, default_max=1030)

        settings["vmin"], settings["vmax"] = max(960, np.floor(vmin/5)*5), min(1050, np.ceil(vmax/5)*5)

        settings["levels"] = np.linspace(settings["vmin"], settings["vmax"], 50)

        settings["contour_levels"] = np.arange(960, 1050, 2)

        settings["contour_colors"], settings["contour_linewidths"], settings["contour_alpha"], settings["extend"], settings["norm"] = 'black', 0.6, 0.7, "both", None



    elif "TEMP" in var_code and ("850" in var_code or "500" in var_code):

        settings["cmap"] = create_temperature_cmap()

        settings["title_tr"] = f"{'850hPa' if '850' in var_code else '500hPa'} Sıcaklık"

        settings["unit"] = "°C"

        

        # GÜVENLİ LİMİT HESAPLAMA (BURASI PATLIYORDU, ARTIK PATLAMAZ)

        default_t = -10 if "850" in var_code else -25

        vmin, vmax = get_safe_limits(data_np, default_min=default_t, default_max=default_t+20)

        

        settings["vmin"], settings["vmax"] = np.floor(vmin/5)*5, np.ceil(vmax/5)*5

        settings["levels"] = np.linspace(settings["vmin"], settings["vmax"], 100)

        settings["contour_levels"] = np.arange(settings["vmin"], settings["vmax"]+1, 2)

        settings["contour_colors"], settings["contour_linewidths"], settings["contour_alpha"], settings["extend"], settings["norm"] = 'black', 0.4, 0.4, "both", None



    elif "WIND" in var_code and ("850" in var_code or "300" in var_code):

        settings["title_tr"] = f"{'850hPa' if '850' in var_code else '300hPa'} Rüzgar"

        settings["unit"] = "knot"

        levels = [20, 30, 40, 50, 60, 70, 80, 100, 120, 150, 180, 200]

        colors = create_wind_cmap()[-len(levels)+1:]

        settings["cmap"] = mcolors.ListedColormap(colors)

        settings["cmap"].set_under('white', alpha=0); settings["cmap"].set_over('#1A0033')

        settings["levels"], settings["norm"], settings["extend"], settings["contour_levels"] = levels, mcolors.BoundaryNorm(levels, len(colors)), "max", None



    return settings



def plot_wind_quivers(ax, u, v, lons, lats):

    u_kt, v_kt = to_np(u) * MS_TO_KT, to_np(v) * MS_TO_KT

    ny, nx = u_kt.shape

    skip = max(1, int(nx / 28))

    ax.quiver(to_np(lons)[::skip, ::skip], to_np(lats)[::skip, ::skip], u_kt[::skip, ::skip], v_kt[::skip, ::skip],

              transform=ccrs.PlateCarree(), scale=400, scale_units='inches', width=0.002, headwidth=3.5, headlength=4.5,

              color='#2C3E50', alpha=0.8, zorder=15)



def add_map_features(ax, domain, res_km):

    ax.add_feature(cfeature.COASTLINE, linewidth=1.0, edgecolor='#2C3E50', zorder=10)

    ax.add_feature(cfeature.BORDERS, linewidth=0.8, edgecolor='#34495E', zorder=10)

    if res_km <= 3:

        ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='#566573', linestyle=':', zorder=9, alpha=0.5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='gray', alpha=0.3, linestyle='--', zorder=8)

    gl.top_labels = gl.right_labels = False

    gl.xlabel_style = gl.ylabel_style = {'size': 7, 'color': '#2C3E50'}



def add_professional_colorbar(fig, ax, cf, settings, domain):

    cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, shrink=0.85, aspect=25, extend=settings.get("extend", "neither"))

    cbar.set_label(f"{settings['title_tr']} [{settings.get('unit','')}]", size=9, weight='normal', labelpad=8)

    if settings.get("norm"):

        cbar.set_ticks(settings["levels"])

        if len(settings["levels"]) > 12: cbar.ax.set_yticklabels([f'{v:.0f}' if i%2==0 else '' for i,v in enumerate(settings["levels"])])

    cbar.ax.tick_params(labelsize=7)

    return cbar



def calculate_heat_index(t2_c, rh):

    t2_f = t2_c * 9/5 + 32

    hi_f = 0.5 * (t2_f + 61.0 + ((t2_f - 68.0) * 1.2) + (rh * 0.094))

    hi_f_full = -42.379 + 2.04901523*t2_f + 10.14333127*rh - 0.22475541*t2_f*rh - 6.83783e-3*t2_f*t2_f - 5.481717e-2*rh*rh + 1.22874e-3*t2_f*t2_f*rh + 8.5282e-4*t2_f*rh*rh - 1.99e-6*t2_f*t2_f*rh*rh

    hi_c = (hi_f_full - 32) * 5/9

    return np.where(t2_c < 20, t2_c, hi_c)



# ==========================================

# 🎯 ANA FONKSİYON (FULL SAĞLAM)

# ==========================================

def generate_maps():

    print("🚀 KEREM PALANCI - Professional Engine v5.0 (ARMORED)")

    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_times = set()



    for dom in DOMAINS:

        res_km = DOMAIN_RES[dom]

        print(f"\n🌍 Domain: {dom.upper()} ({res_km}km)")

        files = sorted(glob.glob(os.path.join(BASE_DIR, f"wrfout_{dom}_*")))

        

        if not files:

            print(f"   ⚠️  Dosya bulunamadı!")

            continue



        for idx, f in enumerate(files):

            try:

                basename = os.path.basename(f)

                dt = datetime.datetime.strptime(basename.replace(f"wrfout_{dom}_", ""), "%Y-%m-%d_%H:%M:%S")

                if ONLY_3H and (dt.hour % 3 != 0): continue

                clean_time = dt.strftime("%Y%m%d_%H%M%S")

                all_times.add(clean_time)

                print(f"   📅 {dt.strftime('%Y-%m-%d %H:%M')} UTC", end=" ")

                

                nc = Dataset(f)

                p_ref = getvar(nc, "T2")  

                lats, lons = latlon_coords(p_ref)

                cart_proj = get_cartopy(p_ref)

                

                var_list = [

                    "T2", "TD2", "RH2", "HEAT_INDEX",

                    "UVMET10", "RAIN", "MDBZ", "SLP",

                    "CAPE", "CIN", "SRH", "SHEAR",

                    "CLOUD", "SNOWH", "SNOWDEPTH",

                    "TEMP_850", "TEMP_500",

                    "WIND_850", "WIND_300"

                ]



                for var_code in var_list:

                    save_path = os.path.join(OUTPUT_DIR, f"{dom}_{var_code}_{clean_time}.png")

                    if os.path.exists(save_path):

                        print(".", end="", flush=True)

                        continue



                    try:

                        u, v = None, None

                        

                        # --- HESAPLAMALAR ---

                        if var_code == "T2":

                            var_data = getvar(nc, "T2") - 273.15

                        elif var_code == "TD2":

                            var_data = getvar(nc, "td2")

                        elif var_code == "RH2":

                            var_data = getvar(nc, "rh2")

                        elif var_code == "UVMET10":

                            u, v = getvar(nc, "uvmet10", units="kt")

                            var_data = getvar(nc, "uvmet10_wspd_wdir")[0] * MS_TO_KT

                        elif var_code == "RAIN":

                            var_data = getvar(nc, "RAINC") + getvar(nc, "RAINNC")

                        elif var_code == "MDBZ":

                            var_data = getvar(nc, "mdbz")

                        elif var_code == "SLP":

                            var_data = getvar(nc, "slp")

                        elif var_code == "CAPE":

                            var_data = getvar(nc, "cape_2d")[0]

                        elif var_code == "CIN":

                            var_data = getvar(nc, "cape_2d")[1]

                        elif var_code == "SRH":

                            var_data = getvar(nc, "srh")

                        

                        elif var_code == "SHEAR":

                            u_3d, v_3d = getvar(nc, "uvmet", units="kt")

                            z = getvar(nc, "z", units="m"); ter = getvar(nc, "ter", units="m")

                            agl = z - ter

                            u_6km = interplevel(u_3d, agl, 6000); v_6km = interplevel(v_3d, agl, 6000)

                            u_10m, v_10m = getvar(nc, "uvmet10", units="kt")

                            var_data = np.sqrt((u_6km - u_10m)**2 + (v_6km - v_10m)**2)



                        elif var_code == "CLOUD":

                            # Fix: Max overlap mantığı

                            cld = getvar(nc, "cloudfrac")

                            var_data = np.max(to_np(cld), axis=0) * 100



                        elif var_code == "SNOWH":

                            var_data = getvar(nc, "SNOWNC")

                        elif var_code == "SNOWDEPTH":

                            var_data = getvar(nc, "SNOWH") * 100



                        elif "TEMP_" in var_code:

                            level = int(var_code.split("_")[1])

                            tc = getvar(nc, "tc")

                            p = getvar(nc, "pressure")

                            var_data = interplevel(tc, p, level)



                        elif "WIND_" in var_code:

                            level = int(var_code.split("_")[1])

                            p = getvar(nc, "pressure")

                            u, v = getvar(nc, "uvmet", units="kt")

                            u = interplevel(u, p, level); v = interplevel(v, p, level)

                            var_data = (u**2 + v**2)**0.5



                        elif var_code == "HEAT_INDEX":

                            t2_np = to_np(getvar(nc, "T2") - 273.15)

                            rh_np = to_np(getvar(nc, "rh2"))

                            var_data = calculate_heat_index(t2_np, rh_np)



                        else: continue



                        # --- ÇİZİM ---

                        data_np = to_np(var_data)

                        

                        # KRİTİK KORUMA: Eğer veri tamamen NaN ise çizme

                        if np.all(np.isnan(data_np)):

                            # print(f"(Veri Boş: {var_code})", end=" ")

                            continue



                        # get_pro_settings artık NaN gelse bile varsayılan değerler döndürüyor

                        settings = get_pro_settings(var_code, data_np, dom)

                        

                        fig = plt.figure(figsize=(12, 10))

                        ax = plt.axes(projection=cart_proj)

                        

                        # Limitleri Referans Değişkenden Al

                        ax.set_xlim(cartopy_xlim(p_ref))

                        ax.set_ylim(cartopy_ylim(p_ref))

                        

                        add_map_features(ax, dom, res_km)

                        

                        if settings["norm"]:

                            cf = ax.contourf(to_np(lons), to_np(lats), data_np, levels=settings["levels"], norm=settings["norm"], cmap=settings["cmap"], transform=ccrs.PlateCarree(), extend=settings["extend"], alpha=1.0)

                        else:

                            cf = ax.contourf(to_np(lons), to_np(lats), data_np, levels=settings["levels"], cmap=settings["cmap"], transform=ccrs.PlateCarree(), extend=settings["extend"], alpha=1.0)

                        

                        if settings["contour_levels"] is not None:

                            try:

                                # Smooth2d bazen NaN kenarlarda hata verebilir, try-except içinde

                                smooth_data = smooth2d(data_np, 3)

                                ct = ax.contour(to_np(lons), to_np(lats), smooth_data, levels=settings["contour_levels"], colors=settings["contour_colors"], linewidths=settings["contour_linewidths"], alpha=settings["contour_alpha"], transform=ccrs.PlateCarree())

                                if var_code == "SLP": ax.clabel(ct, inline=True, fmt='%1.0f', fontsize=8)

                            except:

                                pass



                        if (var_code == "UVMET10" or "WIND_" in var_code) and u is not None:

                            plot_wind_quivers(ax, u, v, lons, lats)



                        add_professional_colorbar(fig, ax, cf, settings, dom)

                        ax.set_title(f"{settings['title_tr']}\nDomain: {dom.upper()} ({res_km}km)", loc='left', fontsize=12, fontweight='bold')

                        ax.set_title(f"Geçerlilik: {dt.strftime('%d.%m.%Y %H:%M')} UTC", loc='right', fontsize=10, color='#E74C3C', fontweight='bold')

                        ax.text(0, -0.04, "Kerem Palancı | WRF Model", transform=ax.transAxes, fontsize=9, color='#555555', va='top', ha='left')



                        plt.savefig(save_path, dpi=120, bbox_inches='tight')

                        plt.close(fig)

                    

                    except Exception as e:

                        print(f"\n❌ HATA ({dom} - {var_code}): {e}")

                        plt.close('all')

                        continue

                    

                    del var_data

                    gc.collect()



                nc.close()

                print(" ✅") 



            except Exception as e:

                print(f"\n   ❌ Dosya Hatası: {e}")

                continue



    with open(os.path.join(OUTPUT_DIR, "file_list.json"), "w") as jf:

        json.dump(list(all_times), jf)



if __name__ == "__main__":

    generate_maps()
