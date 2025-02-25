import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f_bulk(c, eta, W_a, W_b, G_0):
    term1 = W_a * (c**2 * (1 - c)**2)
    sum_eta2 = eta**2
    sum_eta3 = eta**3
    term2 = W_b * (
        c**2 
        + 6 * (1 - c) * sum_eta2 
        - 4 * (2 - c) * sum_eta3 
        + 3 * (sum_eta2)**2
    )
    term3 = G_0 # The constant is obtained form the CALPHAD database
    return term1 + term2 + term3

def main():
    st.title("3D Plot of $f_{\\mathrm{bulk}}$ as a function of $c$ and $\\eta$")
    st.markdown(
        """
        This app plots the function:
        $$
        f_{\\mathrm{bulk}}(c, \\eta) = G_0 W_a \\left[ c^2 (1 - c)^2 \\right] 
        + W_b \\left[ c^2 + 6(1 - c) \\eta^2 - 4(2 - c) \\eta^3 + 3 \\eta^4 \\right]
        $$
        """
    )
    st.markdown(
        """
        The unit of: $$f_{\\mathrm{bulk}}$$ is J/m$$^3$$.
        """
    )
    
    st.sidebar.header("Parameters")
    G_0 = float(st.sidebar.text_input("$G_0$ (e.g., -5.0E6)", "-2.0E6"))
    W_a = float(st.sidebar.text_input("$W_a$ (e.g., 5.0E6)", "1.0E6"))
    W_b = float(st.sidebar.text_input("$W_b$ (e.g., 2.5E6)", "1.0E6"))
    
    st.sidebar.header("Figure Customization")
    label_fontsize = st.sidebar.slider("Label Font Size", 10, 30, 18, 1)
    tick_fontsize = st.sidebar.slider("Tick Font Size", 8, 25, 14, 1)
    grid_thickness = st.sidebar.slider("Grid Thickness", 0.5, 3.0, 1.5, 0.1)
    
    # New sliders for padding
    labelpad = st.sidebar.slider("Axis Label Padding", 5, 40, int(label_fontsize * 0.6), 1)
    tickpad = st.sidebar.slider("Tick Label Padding", 1, 20, int(tick_fontsize * 0.5), 1)
    # New slider for colorbar padding
    cbar_pad = st.sidebar.slider("Colorbar Padding", 0.01, 0.2, 0.1, 0.01)

    colormaps = ["viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "rainbow", "jet"]
    selected_colormap = st.sidebar.selectbox("Select Colormap", colormaps, index=0)
    
    c_values = np.linspace(0, 1, 100)
    eta_values = np.linspace(0, 1, 100)
    c_grid, eta_grid = np.meshgrid(c_values, eta_values)
    f_bulk_values = f_bulk(c_grid, eta_grid, W_a, W_b, G_0)
    
    fig = plt.figure(figsize=(12, 9), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(c_grid, eta_grid, f_bulk_values, cmap=selected_colormap, edgecolor='none')
    
    ax.set_xlabel(r"$c$", fontsize=label_fontsize, fontweight='bold', fontname="serif", labelpad=labelpad)
    ax.set_ylabel(r"$\eta$", fontsize=label_fontsize, fontweight='bold', fontname="serif", labelpad=labelpad)
    ax.set_zlabel(r"$f_{\mathrm{bulk}}$", fontsize=label_fontsize, fontweight='bold', fontname="serif", labelpad=labelpad)
    ax.set_title(r"$f_{\mathrm{bulk}}$ as a function of $c$ and $\eta$", fontsize=label_fontsize + 2, fontweight='bold', fontname="serif", pad=labelpad)
    
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.label.set_rotation(0)
    
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=grid_thickness, pad=tickpad)
    ax.tick_params(axis='z', which='major', labelsize=tick_fontsize, width=grid_thickness, pad=tickpad)
    
    #cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, pad=0.1)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, pad=cbar_pad)
    cbar.set_label(r"$f_{\mathrm{bulk}}$", fontsize=label_fontsize, fontweight='bold', fontname="serif")
    cbar.ax.tick_params(labelsize=tick_fontsize, width=grid_thickness)
    cbar.ax.minorticks_on()
    
    st.pyplot(fig)

if __name__ == "__main__":
    main()

