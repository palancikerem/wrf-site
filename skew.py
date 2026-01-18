#!/usr/bin/env python3
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm
import metpy.calc as mpcalc
from metpy.plots import SkewT, Hodograph
from metpy.units import units
from wrf import getvar, ll_to_xy, to_np
from datetime import datetime
from pathlib import Path
import multiprocessing
import warnings

# ======================
# CONFIG
# ======================
CONFIG = {
    "base_dir": "/home/kerem/WRF_RUN",
    "output_dir": "/home/kerem/wrf_site/images",
    "cities": {
        "Catalca": [41.1444, 28.4611],
        "Besiktas": [41.0428, 29.0075],
        "Goztepe": [40.9764, 29.0572],
        "Istanbul_Merkez": [41.0082, 28.9784],
        "Aydos": [40.9167, 29.2333],
        "Cerkezkoy": [41.2900, 28.0000],
        "Kagithane": [41.0814, 28.9731]  # Kağıthane eklendi
    },
    "num_processes": 8,
    "dpi": 130
}

def process_file_task(task_info):
    f_path, t_idx, dom = task_info
    try:
        nc = Dataset(str(f_path))
        
        # Zaman Etiketi
        times = getvar(nc, "times", timeidx=t_idx)
        dt = datetime.strptime(str(times.values)[:19], "%Y-%m-%dT%H:%M:%S")

        for city, coords in CONFIG["cities"].items():
            x, y = map(int, ll_to_xy(nc, coords[0], coords[1]))

            # DATA (t_idx üzerinden saatlik çekim)
            p = units.hPa * to_np(getvar(nc, "pressure", timeidx=t_idx)[:, y, x])
            t = units.degC * to_np(getvar(nc, "tc", timeidx=t_idx)[:, y, x])
            td = units.degC * to_np(getvar(nc, "td", timeidx=t_idx)[:, y, x])
            z = units.m * to_np(getvar(nc, "z", timeidx=t_idx)[:, y, x])
            omega = getvar(nc, "wa", timeidx=t_idx)[:, y, x] 

            uv = getvar(nc, "uvmet", timeidx=t_idx)
            u = units("m/s") * to_np(uv[0, :, y, x])
            v = units("m/s") * to_np(uv[1, :, y, x])

            # --- TÜM PARAMETRE HESAPLAMALARI ---
            prof = mpcalc.parcel_profile(p, t[0], td[0]).to("degC")
            cape, cin = mpcalc.surface_based_cape_cin(p, t, td)
            li = mpcalc.lifted_index(p, t, prof)[0]
            ki = mpcalc.k_index(p, t, td)
            tt = mpcalc.total_totals_index(p, t, td)
            pw = mpcalc.precipitable_water(p, td)

            i850, i500 = np.argmin(np.abs(p.m - 850)), np.argmin(np.abs(p.m - 500))
            lr_850_500 = -(t[i500].m - t[i850].m) / ((z[i500].m - z[i850].m) / 1000)

            def shear(depth):
                try:
                    su, sv = mpcalc.bulk_shear(p, u, v, height=z, depth=depth * units.m)
                    return np.hypot(su.m, sv.m) * 1.94
                except: return np.nan
            s01, s03, s06 = shear(1000), shear(3000), shear(6000)

            # --- FIGURE LAYOUT ---
            fig = plt.figure(figsize=(18, 11), dpi=CONFIG["dpi"])
            gs = gridspec.GridSpec(2, 3, width_ratios=[0.08, 1.4, 0.8], 
                                   height_ratios=[1, 1.2], 
                                   wspace=0.25, hspace=0.25)

            # OMEGA PANEL
            ax_om = fig.add_subplot(gs[:, 0])
            ax_om.plot(omega, p, color='purple', lw=1.2)
            ax_om.axvline(0, color='k', lw=0.6)
            ax_om.fill_betweenx(p, omega, 0, where=(omega > 0), color='purple', alpha=0.3)
            ax_om.fill_betweenx(p, omega, 0, where=(omega < 0), color='orange', alpha=0.3)
            ax_om.set_yscale('log')
            ax_om.set_ylim(1050, 100)
            ax_om.set_xlim(-0.5, 0.5) 
            ax_om.set_title("W (m/s)", fontsize=9, pad=10)

            # SKEW-T
            skew = SkewT(fig, rotation=45, subplot=gs[:, 1])
            skew.plot(p, t, color="#d62728", lw=2.6, label="Sıcaklık")
            skew.plot(p, td, color="#2ca02c", lw=2.6, label="Çiy Noktası")
            skew.plot(p, prof, "k--", lw=1.4, alpha=0.8)
            skew.shade_cape(p, t, prof, alpha=0.3)
            skew.shade_cin(p, t, prof, td, alpha=0.2)
            skew.plot_dry_adiabats(alpha=0.2, lw=1)
            skew.plot_moist_adiabats(alpha=0.2, lw=1)
            skew.plot_mixing_lines(alpha=0.2, lw=1)

            # WIND BARBS
            spd = mpcalc.wind_speed(u, v).to("knots")
            mask = (p.m <= 1000) & (p.m >= 100)
            skew.plot_barbs(p[mask][::2], u[mask][::2], v[mask][::2], xloc=1.05, length=6, linewidth=0.6)
            skew.ax.set_ylim(1050, 100)
            skew.ax.set_xlim(-40, 45)

            # HODOGRAPH
            ax_h = fig.add_subplot(gs[0, 2])
            h = Hodograph(ax_h, component_range=60)
            h.add_grid(increment=20)
            z_mask = z <= 6000 * units.m
            h.plot_colormapped(u[z_mask], v[z_mask], z[z_mask].m)
            
            # PARAMETRE TABLOSU
            ax_t = fig.add_subplot(gs[1, 2])
            ax_t.axis("off")
            table_data = [
                ["SBCAPE", f"{cape.m:.0f}"], ["SBCIN", f"{cin.m:.0f}"],
                ["LI", f"{li.m:.1f}"], ["PW", f"{pw.m:.1f} mm"],
                ["LR 850-500", f"{lr_850_500:.1f} C/km"],
                ["0-6km Shear", f"{s06:.1f} kt"],
                ["K-Index", f"{ki.m:.0f}"], ["TT-Index", f"{tt.m:.0f}"]
            ]
            tab = ax_t.table(cellText=table_data, loc="center", colWidths=[0.5, 0.4])
            tab.scale(1.2, 1.8)

            # --- BAŞLIK VE ALT YAZILAR ---
            plt.suptitle(f"{city.upper()} ({dom.upper()}) | {dt:%d.%m.%Y %H:%M} UTC", fontsize=18, weight="bold", y=0.98)
            
            # Sol Alt Yazı
            fig.text(0.08, 0.02, "Kerem Palancı | WRF Model", ha='left', fontsize=12, fontweight='bold')
            # Sağ Alt Yazı
            res_info = "3km"
            fig.text(0.92, 0.02, f"WRF {res_info} | {dom.upper()}", ha='right', fontsize=12, fontweight='bold', color='blue')

            out = f"SKEWT_{dom.upper()}_{city.upper()}_{dt:%Y%m%d_%H%M}.png"
            plt.savefig(Path(CONFIG["output_dir"]) / out, bbox_inches="tight")
            plt.close(fig)
        nc.close()
    except Exception as e:
        print(f"Hata: {e}")

def main():
    files = sorted(Path(CONFIG["base_dir"]).glob("wrfout_d02_*"))
    tasks = []
    for f in files:
        nc = Dataset(f, "r")
        num_times = len(nc.dimensions['Time'])
        nc.close()
        for t in range(num_times):
            tasks.append((f, t, "d02"))

    with multiprocessing.Pool(CONFIG["num_processes"]) as p:
        list(tqdm(p.imap(process_file_task, tasks), total=len(tasks)))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()