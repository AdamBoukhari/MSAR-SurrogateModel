import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Création d'une normalisation et d'une colormap
cmap = plt.cm.inferno
norm = mpl.colors.Normalize(vmin=0, vmax=1)

# Création de la figure
fig, ax = plt.subplots(figsize=(2, 6))  # large et étroite pour colorbar verticale

# Création de la colorbar
cbar = mpl.colorbar.ColorbarBase(
    ax,
    cmap=cmap,
    norm=norm,
    orientation='vertical'  # 'horizontal' si tu veux
)

cbar.set_label("Score", fontsize=14)
cbar.ax.tick_params(labelsize=12)

plt.tight_layout()
plt.show()