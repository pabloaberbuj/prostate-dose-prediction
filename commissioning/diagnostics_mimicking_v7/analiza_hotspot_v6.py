import numpy as np

C = r"C:\Users\MEVATE~1\AppData\Local\Temp\claude\c--Pablo-ProstateDoseProject\970561e7-7351-4ce3-9f9f-79b199f3b6ab\scratchpad"
z = np.load(C + r"\valida_v6.npz")
pdrt, aaa, ptv, body = z["pdrt"], z["aaa"], z["ptv"], z["body"]
RX_GY = 78.0
p_pdrt, p_aaa = 100 * pdrt / RX_GY, 100 * aaa / RX_GY

fuera = body & ~ptv
print("=== punto caliente por fuera del PTV (AAA) ===")
print(f"  max fuera PTV: {p_aaa[fuera].max():.1f}%")
vox_cc = 1e-3 * 2.0 * 2.0 * 2.0  # grilla 2mm isotropica
for umbral in (105, 110, 120, 130):
    vol = (p_aaa[fuera] > umbral).sum() * vox_cc
    print(f"  vol >{umbral}% fuera del PTV: {vol:.1f} cm3")

hot = fuera & (p_aaa > 110)
idx = np.argwhere(hot)
print(f"\n  voxels >110% fuera: {hot.sum()} ({hot.sum()*vox_cc:.1f} cm3)")
print(f"  bbox h={idx[:,0].min()}-{idx[:,0].max()}  d={idx[:,1].min()}-{idx[:,1].max()}  w={idx[:,2].min()}-{idx[:,2].max()}")
com_hot = idx.mean(0)
com_ptv = np.argwhere(ptv).mean(0)
print(f"  centro del hotspot (h,d,w) = {com_hot.round(1)}")
print(f"  centro del PTV     (h,d,w) = {com_ptv.round(1)}")
print(f"  diferencia (h,d,w) = {(com_hot-com_ptv).round(1)}  -> mm = {((com_hot-com_ptv)*2.0).round(1)}")

# corte con mas voxels calientes
cnt = hot.reshape(hot.shape[0], -1).sum(1)
z_hot = int(np.argmax(cnt))
print(f"\n  corte h con mas voxels >110% fuera: h={z_hot} ({cnt[z_hot]} vox)")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for h in sorted(set([z_hot, int(com_ptv[0]), idx[:,0].min(), idx[:,0].max()])):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (img, t) in zip(axes, [(p_pdrt[h], "PDRT"), (p_aaa[h], "AAA"), (p_pdrt[h]-p_aaa[h], "PDRT-AAA")]):
        cm = "RdBu_r" if "-" in t else "jet"
        vv = dict(vmin=-40, vmax=40) if "-" in t else dict(vmin=0, vmax=140)
        im = ax.imshow(img, cmap=cm, **vv)
        ax.contour(ptv[h], levels=[0.5], colors="k", linewidths=1.2)
        ax.set_title(f"{t} h={h}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    out = f"{C}\\v6_hotspot_h{h}.png"
    plt.savefig(out, dpi=95)
    plt.close(fig)
    print("guardado", out)
