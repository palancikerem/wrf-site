import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import glob, os, json, datetime, multiprocessing, warnings, gc
from netCDF4 import Dataset
from wrf import (getvar, to_np, latlon_coords, get_cartopy, smooth2d, interplevel)
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import numpy as np
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

BASE_DIR = "/home/kerem/WRF_RUN"
OUTPUT_DIR = "/home/kerem/wrf_site/images"

def create_wind_cmap():
    colors = ['#A020F0', '#7B00FF', '#0000FF', '#007FFF', '#00FFFF','#00FF7F', '#00FF00', '#7FFF00', '#FFFF00', '#FFD700','#FFA500', '#FF7F00', '#FF4500', '#FF0000','#D20000', '#A50000', '#7B0000', '#B03060','#FF1493', '#FF69B4']
    return mcolors.LinearSegmentedColormap.from_list("custom_wind_smooth", colors, N=256)

def create_temperature_cmap():
    neg_colors = ['#2d004b','#4d004b','#542788','#8073ac','#b2abd2','#d8daeb','#08306b','#08519c','#2171b5','#4292c6','#6baed6','#9ecae1','#c6dbef','#deebf7','#f7fbff']
    pos_colors = ['#006400','#008000','#228b22','#32cd32','#7cfc00','#adff2f','#dfff00','#ffff00','#ffd700','#ffcc00','#ffb900','#ffa500','#ff8c00','#ff7f00','#ff4500','#ff0000','#e60000','#cc0000','#990000','#660000']
    return mcolors.ListedColormap(neg_colors + pos_colors)

def create_rain_cmap():
    colors_hex = ['#dedef2','#b4d7ff','#75baff','#359aff','#0482ff','#0069d2','#00367f','#148f1b','#1acf05','#63ed07','#fff42b','#e8dc00','#f06000','#ff7f27','#ffa66a','#f84e78','#f71e54','#db0f2a','#a30000','#880000','#64007f','#c200fb','#dd66ff','#eba6ff','#f9e6ff']
    return mcolors.ListedColormap(colors_hex)

def create_snow_ptype_cmap():
    return mcolors.LinearSegmentedColormap.from_list("snow_ptype", ["#ffccff", "#ff00ff", "#800080"], N=256)

def create_rain_ptype_cmap():
    return mcolors.LinearSegmentedColormap.from_list("rain_ptype", ["#ccffcc", "#00ff00", "#006400"], N=256)

def get_pro_settings(var_code, dom=None):
    settings = {"extend": "max", "norm": None, "contour_levels": None}
    if var_code in ["T2", "TEMP_850", "TEMP_500", "HEAT_INDEX", "TD2"]:
        settings.update({"levels": np.arange(-30, 43, 1), "cmap": create_temperature_cmap(), "unit": "°C", "extend": "both"})
        if var_code in ["TEMP_850", "TEMP_500"]: settings["contour_levels"] = np.arange(-40, 45, 2)
        titles = {"T2":"2m Sıcaklık", "TEMP_850":"850hPa Sıcaklık", "TEMP_500":"500hPa Sıcaklık", "HEAT_INDEX":"Hissedilen Sıcaklık", "TD2":"2m Çiy Noktası"}
        settings["title_tr"] = titles.get(var_code)
    elif var_code in ["RAIN","RAIN1H","SNOW","SNOW1H","SNOWDEPTH"]:
        settings["cmap"] = create_rain_cmap()
        if var_code == "RAIN": settings.update({"levels":[0.1,1,2,5,10,15,20,25,30,35,40,45,50,60,70,80,90,100,125,150,175,200,250,300,400,500],"title_tr":"Toplam Yağış"})
        elif var_code == "RAIN1H": settings.update({"levels":[0.1,0.2,0.5,1,2,3,4,5,7,10,15,20,25,30,40,50,60],"title_tr":"Saatlik Yağış"})
        elif var_code == "SNOW": settings.update({"levels":[0.1,0.5,1,2,3,4,5,6,8,10,12,14,16,18,20,22,24,26,28,30],"title_tr":"Toplam Kar (cm)"})
        elif var_code == "SNOW1H": settings.update({"levels":[0.1,0.5,1,2,3,4,5,7,10,15,20,25,30],"title_tr":"Saatlik Kar (cm)"})
        elif var_code == "SNOWDEPTH":
            if dom == "d02": settings.update({"levels":[0.5,1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,22,24,26,28,30],"title_tr":"Kar Derinliği (cm)"})
            else: settings.update({"levels":[1,2,5,10,15,20,25,30,40,50,60,75,100,150],"title_tr":"Kar Derinliği (cm)"})
        settings["norm"] = mcolors.BoundaryNorm(settings["levels"], settings["cmap"].N)
    elif var_code == "PTYPE":
        settings.update({"levels": [0.1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20], "title_tr": "Yağış Türü ve Şiddeti", "unit": "mm-cm/h"})
    elif var_code in ["UVMET10","WIND_850","WIND_300"]:
        settings.update({"cmap": create_wind_cmap(), "unit": "km/h", "extend": "both"})
        if var_code == "UVMET10": settings.update({"levels": np.arange(2, 151, 2), "title_tr": "10m Hamle"})
        elif var_code == "WIND_850": settings.update({"levels": np.arange(2, 161, 4), "title_tr": "850hPa Rüzgar"})
        elif var_code == "WIND_300": settings.update({"levels": np.arange(40, 301, 5), "title_tr": "300hPa Rüzgar"})
    elif var_code == "CAPE": settings.update({"levels":[100,250,500,750,1000,1500,2000,2500,3000,4000],"cmap":plt.cm.YlOrRd,"title_tr":"CAPE","unit":"J/kg"})
    elif var_code == "MDBZ": settings.update({"levels":[5,10,15,20,25,30,35,40,45,50,55,60],"cmap":plt.cm.jet,"title_tr":"Radar","unit":"dBZ"})
    elif var_code == "SLP": settings.update({"levels": np.arange(980,1041,2), "cmap": plt.cm.RdYlBu_r, "title_tr": "Basınç", "unit": "hPa", "contour_levels": np.arange(960,1050,4), "extend": "both"})
    elif var_code == "CLOUD": settings.update({"levels":np.arange(10,101,10),"cmap":plt.cm.Greys,"title_tr":"Bulutluluk","unit":"%"})
    return settings

def process_single_file(file_info):
    f, dom, prev_f = file_info
    if not os.path.exists(f) or os.path.getsize(f) == 0: return None
    results_summary = []
    try:
        nc = Dataset(f, "r")
        num_times = len(nc.dimensions['Time'])
        nc_prev = None
        if prev_f and os.path.exists(prev_f):
            nc_prev = Dataset(prev_f, "r")

        try:
            shapefile = shpreader.natural_earth(resolution='10m',category='cultural',name='admin_1_states_provinces')
            provinces = list(shpreader.Reader(shapefile).records())
        except: provinces = []

        for t_idx in range(num_times):
            times = getvar(nc, "times", timeidx=t_idx)
            dt = datetime.datetime.strptime(str(times.values)[:19], "%Y-%m-%dT%H:%M:%S")
            clean_time = dt.strftime("%Y%m%d_%H%M%S")
            p_ref = getvar(nc, "T2", timeidx=t_idx)
            lats, lons = latlon_coords(p_ref)
            cart_proj = get_cartopy(p_ref)
            var_list = ["T2","TD2","TEMP_850","TEMP_500","HEAT_INDEX","RAIN","RAIN1H","SNOW","SNOW1H","SNOWDEPTH","PTYPE","UVMET10","WIND_850","WIND_300","CAPE","MDBZ","SLP","CLOUD"]

            for var_code in var_list:
                try:
                    settings = get_pro_settings(var_code, dom)
                    var_data = u = v = rain_data = snow_data = None
                    if var_code == "T2": var_data = getvar(nc,"T2", timeidx=t_idx) - 273.15
                    elif var_code == "TD2": var_data = getvar(nc,"td2", timeidx=t_idx)
                    elif var_code == "TEMP_850":
                        p = getvar(nc,"pressure", timeidx=t_idx); tc = getvar(nc,"tc", timeidx=t_idx)
                        var_data = interplevel(tc,p,850.0)
                    elif var_code == "TEMP_500":
                        p = getvar(nc,"pressure", timeidx=t_idx); tc = getvar(nc,"tc", timeidx=t_idx)
                        var_data = interplevel(tc,p,500.0)
                    elif var_code == "HEAT_INDEX":
                        t_c = getvar(nc,"T2", timeidx=t_idx) - 273.15; rh = getvar(nc,"rh2", timeidx=t_idx)
                        var_data = t_c + 0.5555*(6.11*np.exp(5417.7530*(1/273.16-1/(t_c+273.15)))*(rh/100)-10)
                    elif var_code in ["RAIN", "RAIN1H", "SNOW", "SNOW1H", "PTYPE"]:
                        curr_r = np.array(nc.variables['RAINC'][t_idx] + nc.variables['RAINNC'][t_idx])
                        curr_s = np.array(nc.variables['SNOWNC'][t_idx])
                        if nc_prev is not None:
                            prev_r = np.array(nc_prev.variables['RAINC'][-1] + nc_prev.variables['RAINNC'][-1])
                            prev_s = np.array(nc_prev.variables['SNOWNC'][-1])
                            diff_r = curr_r - prev_r; diff_s = curr_s - prev_s
                        else:
                            diff_r = curr_r; diff_s = curr_s
                        diff_r[diff_r < 0] = 0; diff_s[diff_s < 0] = 0
                        if var_code == "RAIN": var_data = curr_r
                        elif var_code == "RAIN1H": var_data = diff_r
                        elif var_code == "SNOW": var_data = curr_s
                        elif var_code == "SNOW1H": var_data = diff_s
                        elif var_code == "PTYPE": rain_data = diff_r; snow_data = diff_s; var_data = rain_data
                    elif var_code == "SNOWDEPTH": var_data = getvar(nc,"SNOWH", timeidx=t_idx)*100.0
                    elif var_code == "UVMET10": u,v = getvar(nc,"uvmet10", timeidx=t_idx); var_data = np.sqrt(u**2+v**2)*3.6
                    elif var_code == "WIND_850":
                        p = getvar(nc,"pressure", timeidx=t_idx); uv = getvar(nc,"uvmet", timeidx=t_idx)
                        u = interplevel(uv[0],p,850.0); v = interplevel(uv[1],p,850.0); var_data = np.sqrt(u**2+v**2)*3.6
                    elif var_code == "WIND_300":
                        p = getvar(nc,"pressure", timeidx=t_idx); uv = getvar(nc,"uvmet", timeidx=t_idx)
                        u = interplevel(uv[0],p,300.0); v = interplevel(uv[1],p,300.0); var_data = np.sqrt(u**2+v**2)*3.6
                    elif var_code == "CAPE": var_data = getvar(nc,"cape_2d", timeidx=t_idx)[0]
                    elif var_code == "MDBZ": var_data = getvar(nc,"mdbz", timeidx=t_idx)
                    elif var_code == "SLP": var_data = getvar(nc,"slp", timeidx=t_idx)
                    elif var_code == "CLOUD": var_data = getvar(nc,"cloudfrac", timeidx=t_idx)[0] * 100

                    if var_data is None and rain_data is None: continue
                    save_name = f"{dom}_{var_code}_{clean_time}.webp"
                    save_path = os.path.join(OUTPUT_DIR, save_name)
                    fig = plt.figure(figsize=(12,10))
                    ax = plt.axes(projection=cart_proj)
                    ax.add_feature(cfeature.COASTLINE,linewidth=1); ax.add_feature(cfeature.BORDERS,linewidth=0.8)
                    for prov in provinces:
                        if prov.attributes.get('admin')=='Turkey':
                            ax.add_geometries([prov.geometry],ccrs.PlateCarree(), facecolor='none',edgecolor='black', linewidth=0.4,alpha=0.3)
                    
                    if var_code == "PTYPE":
                        levs = settings.get("levels")
                        cf_r = ax.contourf(to_np(lons), to_np(lats), to_np(rain_data), levels=levs, cmap=create_rain_ptype_cmap(), extend="max", transform=ccrs.PlateCarree(), alpha=0.8)
                        cf_s = ax.contourf(to_np(lons), to_np(lats), to_np(snow_data), levels=levs, cmap=create_snow_ptype_cmap(), extend="max", transform=ccrs.PlateCarree(), alpha=0.8)
                        
                        # Skalaları harita alanının dışına (sağa) taşıma
                        ticks_val = [2, 5, 10, 15, 20]
                        cbar_ax_s = fig.add_axes([0.92, 0.52, 0.015, 0.35]) # Sağdaki kar skalası
                        plt.colorbar(cf_s, cax=cbar_ax_s, label="Kar (cm/h)", ticks=ticks_val)
                        cbar_ax_r = fig.add_axes([0.92, 0.12, 0.015, 0.35]) # Sağdaki yağmur skalası
                        plt.colorbar(cf_r, cax=cbar_ax_r, label="Yağmur (mm/h)", ticks=ticks_val)
                    else:
                        cf = ax.contourf(to_np(lons),to_np(lats),to_np(var_data), levels=settings.get("levels"), cmap=settings.get("cmap"), norm=settings.get("norm"), extend=settings.get("extend"), transform=ccrs.PlateCarree())
                        # Standart skalayı da içeri girmemesi için pad ile sağa çektim
                        plt.colorbar(cf, ax=ax, shrink=0.75, pad=0.08)

                    if u is not None: ax.streamplot(to_np(lons),to_np(lats),to_np(u),to_np(v), transform=ccrs.PlateCarree(), color='white',linewidth=0.6,density=2)
                    if settings.get("contour_levels") is not None:
                        cs = ax.contour(to_np(lons),to_np(lats),smooth2d(to_np(var_data),3), levels=settings["contour_levels"], colors='black',linewidths=0.6, transform=ccrs.PlateCarree())
                        ax.clabel(cs,fontsize=8)
                    res_info = "9km" if dom == "d01" else "3km"
                    ax.set_title(f"{dom.upper()} ({res_info}) - {settings['title_tr']}",loc='left',fontweight='bold')
                    ax.set_title(f"UTC {dt.strftime('%d.%m %H:%M')}Z",loc='right',color='red',fontweight='bold')
                    ax.text(0.99,-0.04,f"WRF {res_info} | {dom.upper()}",transform=ax.transAxes,ha='right',fontweight='bold',color='blue')
                    ax.text(0.01,-0.04,"Kerem Palancı | WRF Model",transform=ax.transAxes,ha='left',fontweight='bold')
                    plt.savefig(save_path,dpi=110,bbox_inches='tight',pil_kwargs={'quality':90,'method':6}); plt.close(fig)
                except Exception as e:
                    print(f"Hata {dom}-{var_code} Saat {clean_time}: {e}"); plt.close('all')
            results_summary.append({"dom": dom, "time": clean_time})
        
        nc.close()
        if nc_prev: nc_prev.close()
    except Exception as e: print(f"Dosya Hatası ({f}): {e}")
    gc.collect()
    return results_summary

def generate_maps():
    print("Görselleşiyor...")
    os.makedirs(OUTPUT_DIR,exist_ok=True)
    all_files_info = []
    for dom in ["d01", "d02"]:
        files = sorted(glob.glob(os.path.join(BASE_DIR,f"wrfout_{dom}_*")))
        for i, f in enumerate(files):
            prev_f = files[i-1] if i > 0 else None
            all_files_info.append((f, dom, prev_f))
    with multiprocessing.Pool(max(1,multiprocessing.cpu_count()-1)) as pool:
        results = pool.map(process_single_file, all_files_info)
    file_registry = {"d01": [], "d02": []}
    for r_list in results:
        if r_list:
            for r in r_list: file_registry[r["dom"]].append(r["time"])
    for d in file_registry: file_registry[d] = sorted(list(set(file_registry[d])))
    with open(os.path.join(OUTPUT_DIR,"file_list.json"),"w") as jf:
        json.dump(file_registry, jf, indent=2)
    print("Bitti")

if __name__ == "__main__":
    generate_maps()