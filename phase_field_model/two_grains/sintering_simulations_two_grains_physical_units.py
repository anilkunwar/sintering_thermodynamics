import streamlit as st
import numpy as np
import plotly.express as px
from scipy.ndimage import convolve
import pyvista as pv
import os
import io
import time
from pathlib import Path
import shutil

# Physical scales
L0 = 10e-6  # meters (10 µm)
T0 = 100  # seconds
f0 = 1e6  # J/m³
M0 = 1e-14  # m⁵/(J·s) (mobility scale)

# Streamlit app configuration
st.title("2D Phase-Field Sintering Simulation (Physical Units)")
st.write("Finite difference method to solve PDEs for sintering of powder grains (diameters ~10–40 µm) over 1–10 s.")

# Initialize session state for geometry
if "geometry_confirmed" not in st.session_state:
    st.session_state.geometry_confirmed = False
if "N" not in st.session_state:
    st.session_state.N = 5
if "centers" not in st.session_state:
    st.session_state.centers = []
if "radii" not in st.session_state:
    st.session_state.radii = []

# Sidebar for parameters
st.sidebar.header("Simulation Parameters (Physical Units)")
nx = st.sidebar.slider("Grid Size (nx)", 50, 150, 100, step=10)
ny = st.sidebar.slider("Grid Size (ny)", 50, 150, 100, step=10)
dx = st.sidebar.slider("Grid Spacing (dx, µm)", 0.5, 2.0, 1.0, step=0.1) * 1e-6  # Convert µm to m
dt = st.sidebar.slider("Time Step (dt, s)", 1e-5, 1e-3, 1e-4, step=1e-5, format="%.1e")  # Seconds
total_time = st.sidebar.slider("Total Simulation Time (s)", 1.0, 10.0, 5.0, step=0.1)  # Seconds
output_interval = 1.0  # Seconds (0.01 nondimensional × 100 s)

# Phase-field parameters (physical units)
A = st.sidebar.slider("Free Energy Constant A (MJ/m³)", 10.0, 20.0, 16.0) * 1e6  # J/m³
B = st.sidebar.slider("Free Energy Constant B (MJ/m³)", 0.5, 3.0, 2.0) * 1e6  # J/m³
kappa_c = st.sidebar.slider("Gradient Coefficient κ_c (J/m)", 0.3, 1.2, 0.5)  # J/m
kappa_eta = st.sidebar.slider("Gradient Coefficient κ_η (J/m)", 0.03, 0.07, 0.05)  # J/m
M_s = st.sidebar.slider("Surface Mobility M_s (m⁵/(J·s))", 1e-22, 95e-22, 10e-22)  # m⁵/(J·s)
M_gb = st.sidebar.slider("Grain Boundary Mobility M_gb (m⁵/(J·s))", 15e-22, 55e-22, 20e-22)  # m⁵/(J·s)
M_b = st.sidebar.slider("Bulk Mobility M_b (m⁵/(J·s))", 1e-22, 95e-22, 1e-22)  # m⁵/(J·s)
M_v = st.sidebar.slider("Vapor Mobility M_v (m⁵/(J·s))", 0.5e-22, 2e-22, 1e-22)  # m⁵/(J·s)
mobility_exponent = st.sidebar.slider("Mobility Exponent (log₁₀ α)", -7.0, -5.0, -6.0, step=0.1)
alpha = 10 ** mobility_exponent  # Dimensionless scaling
L = st.sidebar.slider("Grain Boundary Mobility L (s⁻¹)", 0.5, 1.5, 1.0)  # s⁻¹
k = st.sidebar.slider("Stiffness Constant k (MJ/m³)", 0.1, 0.2, 0.14) * 1e6  # J/m³
m_t = st.sidebar.slider("Translational Mobility m_t (m⁵/(J·s))", 1e-3, 3e-3, 2e-3)  # m⁵/(J·s)
c_0 = st.sidebar.slider("Equilibrium Density c_0", 0.9, 1.0, 0.9816)

# User-defined number of particles
N = st.sidebar.slider("Number of Powder Particles (N)", 1, 10, 2)
st.session_state.N = N

# Geometry input
st.sidebar.header(f"Particle Geometry (η_2 to η_{N+1})")
x_max = nx * dx  # meters
y_max = ny * dx  # meters
default_centers_frac = [(0.25, 0.75), (0.75, 0.75), (0.75, 0.25), (0.25, 0.25), (0.5, 0.5)] + [(0.5, 0.5)] * 5
default_radii = [0.2 * (x_max * 1e6), 0.15 * (x_max * 1e6), 0.1 * (x_max * 1e6), 0.08 * (x_max * 1e6), 0.12 * (x_max * 1e6)] + [0.1 * (x_max * 1e6)] * 5  # µm
max_radius = (x_max + y_max) * 0.5 * 1e6  # µm

centers = []
radii = []
for i in range(N):
    st.sidebar.subheader(f"Particle η_{i+2}")
    cx_frac = st.sidebar.number_input(f"Center x (fraction of x_max, η_{i+2})", min_value=0.0, max_value=1.0,
                                     value=float(default_centers_frac[i][0]), step=0.01, key=f"cx_{i}")
    cy_frac = st.sidebar.number_input(f"Center y (fraction of y_max, η_{i+2})", min_value=0.0, max_value=1.0,
                                     value=float(default_centers_frac[i][1]), step=0.01, key=f"cy_{i}")
    cx = cx_frac * x_max  # meters
    cy = cy_frac * y_max  # meters
    r = st.sidebar.slider(f"Radius (η_{i+2}, µm)", min_value=0.05 * (x_max * 1e6), max_value=max_radius,
                          value=default_radii[i], step=0.1, key=f"r_{i}") * 1e-6  # Convert µm to m
    centers.append((cx, cy))
    radii.append(r)

# Preview initial density field
def generate_preview_c(nx, ny, dx, centers, radii, N):
    c = np.zeros((nx, ny))
    eta = [np.zeros((nx, ny)) for _ in range(N+1)]
    delta = 5e-6  # Interface width, 5 µm
    x, y = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dx)
    for p, (cx, cy) in enumerate(centers):
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        eta[p+1] = 0.5 * (1 - np.tanh((r - radii[p]) / delta))
    c = np.clip(sum(eta[1:]), 0, 1)
    return c

st.subheader("Geometry Preview (c = Σ η_i for i=2 to N+1)")
preview_c = generate_preview_c(nx, ny, dx, centers, radii, N)
fig_preview = px.imshow(
    preview_c,
    color_continuous_scale="Viridis",
    title="Initial Density Field (c)",
    labels={"x": "x (µm)", "y": "y (µm)"},
    zmin=0,
    zmax=1
)
fig_preview.update_layout(width=400, height=400)
st.plotly_chart(fig_preview, use_container_width=True)

# Geometry confirmation
if st.sidebar.button("Completed Geometry"):
    st.session_state.geometry_confirmed = True
    st.session_state.centers = centers
    st.session_state.radii = radii
    st.sidebar.success("Geometry confirmed! Click 'Run Simulation' to proceed.")

# Create temporary directory
output_dir = Path("sintering_outputs")
output_dir.mkdir(exist_ok=True)

# Cache simulation
@st.cache_data
def run_simulation(nx, ny, dx, dt, total_time, output_interval, A, B, kappa_c, kappa_eta,
                  M_s, M_gb, M_b, M_v, alpha, L, k, m_t, c_0, N, centers, radii):
    n_steps = int(total_time / dt)
    output_steps = int(output_interval / dt)

    # Laplacian kernel
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]]) / dx**2  # m⁻²

    def h(eta, i):
        eta2_sum = sum(eta_j**2 for eta_j in eta)
        return np.where(eta2_sum > 1e-10, eta[i]**2 / eta2_sum, 0.0)

    def initialize_microstructure(nx, ny, dx, centers, radii, N):
        c = np.zeros((nx, ny))
        eta = [np.zeros((nx, ny)) for _ in range(N+1)]
        delta = 5e-6  # 5 µm
        x, y = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dx)
        for p, (cx, cy) in enumerate(centers):
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            eta[p+1] = 0.5 * (1 - np.tanh((r - radii[p]) / delta))
        c = np.clip(sum(eta[1:]), 0, 1)
        eta[0] = np.clip(1 - c, 0, 1)
        return c, eta, centers, radii

    def free_energy_density(c, eta):
        sum_eta2 = sum(eta_i**2 for eta_i in eta[1:])
        sum_eta3 = sum(eta_i**3 for eta_i in eta[1:])
        f_vapor = A * c**2 * (1 - c)**2  # J/m³
        f_powder = B * (c**2 + 6 * (1 - c) * sum_eta2 - 4 * (2 - c) * sum_eta3 + 3 * sum_eta2**2)
        return f_vapor + 5 * f_powder  # J/m³

    def compute_total_free_energy(c, eta):
        f = free_energy_density(c, eta)
        grad_c = np.gradient(c, dx, dx, axis=(0, 1))
        grad_energy_c = (kappa_c / 2) * (grad_c[0]**2 + grad_c[1]**2)  # J/m³
        grad_energy_eta = sum((kappa_eta / 2) * (np.gradient(eta_i, dx, dx, axis=(0, 1))[0]**2 +
                               np.gradient(eta_i, dx, dx, axis=(0, 1))[1]**2) for eta_i in eta)
        F = np.sum(f + grad_energy_c + grad_energy_eta) * dx**2  # J
        return F

    def compute_postprocessed(c, eta):
        h_i = [h(eta, i) for i in range(N+1)]
        c_h_i2 = sum(c * h_i_j**2 for h_i_j in h_i[1:])
        eta_i2 = sum(eta_i**2 for eta_i in eta[1:])
        f = free_energy_density(c, eta)
        f_h_i = [f * h_i_j for h_i_j in h_i]
        return c_h_i2, eta_i2, f_h_i

    def plot_free_energy_vs_c(c, eta, nx, ny):
        mid_y = ny // 2
        c_line = c[:, mid_y]
        eta_mid = [eta_i[:, mid_y] for eta_i in eta]
        f_line = free_energy_density(c_line, eta_mid) / 1e6  # MJ/m³
        fig = px.scatter(x=c_line, y=f_line, labels={"x": "c", "y": "Free Energy Density (MJ/m³)"},
                         title="Free Energy Density vs. c (Middle Row)")
        return fig

    def mobility(c, eta):
        sum_eta_ij = sum(eta[i] * eta[j] for i in range(1, N+1) for j in range(1, N+1) if i != j)
        h_vapor = h(eta, 0)
        h_solid = sum(h(eta, i) for i in range(1, N+1))
        M = alpha * (M_v * h_vapor + M_b * h_solid + M_s * c**2 * (1 - c)**2 + M_gb * sum_eta_ij)
        return M  # m⁵/(J·s)

    def laplacian(field):
        return convolve(field, kernel, mode='wrap')  # m⁻²

    def dF_dc(c, eta):
        sum_eta2 = sum(eta_i**2 for eta_i in eta[1:])
        sum_eta3 = sum(eta_i**3 for eta_i in eta[1:])
        df_dc_vapor = A * 2 * c * (1 - c) * (1 - 2 * c)  # J/m³
        df_dc_powder = 2 * B * (2 * c - 6 * sum_eta2 + 4 * sum_eta3)
        grad_term = -kappa_c * laplacian(c)  # J/m³
        return df_dc_vapor + df_dc_powder + grad_term

    def dF_deta_i(eta_i, c, eta, i):
        if i == 0:
            return 0
        sum_eta2 = sum(eta_j**2 for eta_j in eta[1:])
        df_deta_i = 2 * B * (12 * (1 - c) * eta_i - 12 * (2 - c) * eta_i**2 + 12 * sum_eta2 * eta_i)
        grad_term = -kappa_eta * laplacian(eta_i)
        return df_deta_i + grad_term  # J/m³

    def compute_velocity(c, eta, centers, radii):
        velocities = []
        x, y = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dx)
        for p in range(N):
            cx, cy = centers[p]
            r_eff = radii[p]
            r_cut = 2.2 * r_eff
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            mask = dist <= r_cut
            eta_idx = p + 1

            force = np.zeros(2)
            volume = np.sum(eta[eta_idx][mask] * dx**2)  # m²
            c_ij_sum = np.zeros_like(c)
            eta_diff = np.zeros_like(c)
            for q in range(1, N+1):
                if q != eta_idx:
                    c_ij = eta[eta_idx] * eta[q]
                    c_ij_sum += np.where(c_ij > 0, c_ij, 0)
                    eta_diff += np.where(c_ij > 0, eta[eta_idx] - eta[q], 0)
            dF_i = k * (c - c_0) * c_ij_sum * eta_diff  # J/m³
            force[0] = np.sum(dF_i[mask] * dx**2)  # J/m
            force[1] = np.sum(dF_i[mask] * dx**2)
            v_t = (m_t / volume) * force if volume > 0 else np.zeros(2)  # m/s
            velocities.append(v_t)
        return velocities

    def advection_term(field, v):
        grad_x = (field - np.roll(field, 1, axis=0)) / dx if v[0] >= 0 else (np.roll(field, -1, axis=0) - field) / dx
        grad_y = (field - np.roll(field, 1, axis=1)) / dx if v[1] >= 0 else (np.roll(field, -1, axis=1) - field) / dx
        div_v = 0.0
        return v[0] * grad_x + v[1] * grad_y + field * div_v  # s⁻¹

    def save_vts(c, eta, c_h_i2, eta_i2, f_h_i, t, output_dir, step):
        x = np.arange(nx) * dx * 1e6  # µm
        y = np.arange(ny) * dx * 1e6  # µm
        z = np.array([0.0])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)

        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = (nx, ny, 1)

        grid.point_data["c"] = c.flatten(order='F')
        for i in range(N+1):
            grid.point_data[f"eta_{i+1}"] = eta[i].flatten(order='F')
        grid.point_data["c_h_i2"] = c_h_i2.flatten(order='F')
        grid.point_data["eta_i2"] = eta_i2.flatten(order='F')
        for i in range(N+1):
            grid.point_data[f"f_h_{i+1}"] = f_h_i[i].flatten(order='F') / 1e6  # MJ/m³

        filename = output_dir / f"sintering_t{t:.3f}.vts"
        grid.save(filename)
        return filename

    # Main simulation
    c, eta, centers, radii = initialize_microstructure(nx, ny, dx, centers, radii, N)
    t = 0.0
    outputs = []
    free_energies = []
    next_output_time = output_interval

    progress_bar = st.progress(0)
    status_text = st.empty()

    for step in range(n_steps):
        mu = dF_dc(c, eta)
        dF_deta = [dF_deta_i(eta_i, c, eta, i) for i, eta_i in enumerate(eta)]
        M = mobility(c, eta)
        velocities = compute_velocity(c, eta, centers, radii)

        # Update c
        lap_mu = laplacian(mu)
        advection = np.zeros_like(c)
        for p in range(N):
            eta_idx = p + 1
            r_eff = radii[p]
            r_cut = 2.2 * r_eff
            x, y = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dx)
            cx, cy = centers[p]
            mask = np.sqrt((x - cx)**2 + (y - cy)**2) <= r_cut
            adv_term = advection_term(c, velocities[p])
            advection += np.where(mask, adv_term * eta[eta_idx], 0)
        dc_dt = laplacian(M * lap_mu) - advection
        c += dt * dc_dt
        c = np.clip(c, 0, 1)

        # Update eta_i
        for i in range(1, N+1):
            advection = np.zeros_like(eta[i])
            for p in range(N):
                if i == p + 1:
                    r_eff = radii[p]
                    r_cut = 2.2 * r_eff
                    x, y = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dx)
                    cx, cy = centers[p]
                    mask = np.sqrt((x - cx)**2 + (y - cy)**2) <= r_cut
                    adv_term = advection_term(eta[i], velocities[p])
                    advection += np.where(mask, adv_term, 0)
            deta_dt = -L * dF_deta[i] - advection
            eta[i] += dt * deta_dt
            eta[i] = np.clip(eta[i], 0, 1)

        # Update eta_1
        eta[0] = np.clip(1 - sum(eta[1:]), 0, 1)

        # Update centers
        for p, v in enumerate(velocities):
            centers[p] = (centers[p][0] + v[0] * dt, centers[p][1] + v[1] * dt)

        t += dt

        # Compute total free energy
        F_total = compute_total_free_energy(c, eta)
        free_energies.append((t, F_total))

        # Output at intervals
        if t >= next_output_time or step == n_steps - 1:
            c_h_i2, eta_i2, f_h_i = compute_postprocessed(c, eta)
            try:
                vts_file = save_vts(c, eta, c_h_i2, eta_i2, f_h_i, t, output_dir, step)
            except Exception as e:
                st.error(f"Failed to save VTS file at t={t:.3f} s: {str(e)}")
                continue

            fields = {
                "c": c,
                "eta_1": eta[0],
                "eta_2": eta[1],
                "eta_3": eta[2] if N >= 2 else np.zeros_like(c),
                "eta_4": eta[3] if N >= 3 else np.zeros_like(c),
                "eta_5": eta[4] if N >= 4 else np.zeros_like(c),
                "c_h_i2": c_h_i2,
                "eta_i2": eta_i2
            }
            plot_files = {}
            for name, field in fields.items():
                fig = px.imshow(
                    field,
                    color_continuous_scale="Viridis",
                    title=f"{name} at t={t:.3f} s",
                    labels={"x": "x (µm)", "y": "y (µm)"},
                    zmin=0,
                    zmax=1 if name not in ["c_h_i2", "eta_i2"] else None
                )
                fig.update_layout(width=350, height=350)
                plot_file = output_dir / f"{name}_t{t:.3f}.png"
                try:
                    fig.write_image(str(plot_file), engine="kaleido")
                except Exception as e:
                    st.warning(f"Failed to save Plotly image for {name} at t={t:.3f} s: {str(e)}")
                plot_files[name] = (fig, plot_file)

            for i in range(min(N+1, 5)):
                name = f"f_h_{i+1}"
                field = f_h_i[i] / 1e6  # MJ/m³
                fig = px.imshow(
                    field,
                    color_continuous_scale="Viridis",
                    title=f"f_h_{i+1} at t={t:.3f} s",
                    labels={"x": "x (µm)", "y": "y (µm)"}
                )
                fig.update_layout(width=350, height=350)
                plot_file = output_dir / f"{name}_t{t:.3f}.png"
                try:
                    fig.write_image(str(plot_file), engine="kaleido")
                except Exception as e:
                    st.warning(f"Failed to save Plotly image for {name} at t={t:.3f} s: {str(e)}")
                plot_files[name] = (fig, plot_file)

            fig_energy = plot_free_energy_vs_c(c, eta, nx, ny)
            energy_plot_file = output_dir / f"free_energy_vs_c_t{t:.3f}.png"
            try:
                fig_energy.write_image(str(energy_plot_file), engine="kaleido")
            except Exception as e:
                st.warning(f"Failed to save free energy plot at t={t:.3f} s: {str(e)}")
            plot_files["free_energy_vs_c"] = (fig_energy, energy_plot_file)

            outputs.append((t, vts_file, plot_files))
            next_output_time += output_interval

        progress_bar.progress(min(step / n_steps, 1.0))
        status_text.text(f"Simulating... Step {step}/{n_steps}, Time {t:.3f} s")

    return outputs, free_energies

# Run simulation
if st.button("Run Simulation"):
    if not st.session_state.geometry_confirmed:
        st.error("Please click 'Completed Geometry' to confirm particle geometry before running the simulation.")
    else:
        with st.spinner("Running simulation..."):
            try:
                outputs, free_energies = run_simulation(nx, ny, dx, dt, total_time, output_interval, A, B, kappa_c, kappa_eta,
                                                        M_s, M_gb, M_b, M_v, alpha, L, k, m_t, c_0, N,
                                                        st.session_state.centers, st.session_state.radii)
            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")
                st.stop()
        
        st.success("Simulation complete!")
        
        st.header("Simulation Results")
        for t, vts_file, plot_files in outputs:
            st.subheader(f"Time t={t:.3f} s")
            
            st.write("**Density and Phase Fields**")
            cols = st.columns(4)
            for i, name in enumerate(["c", "eta_1", "eta_2", "eta_3"]):
                if name in plot_files:
                    with cols[i]:
                        st.plotly_chart(plot_files[name][0], use_container_width=True)
            
            if N >= 3:
                st.write("**Additional Phase Fields and Postprocessed Variables**")
                cols = st.columns(4)
                for i, name in enumerate(["eta_4", "eta_5", "c_h_i2", "eta_i2"]):
                    if name in plot_files:
                        with cols[i]:
                            st.plotly_chart(plot_files[name][0], use_container_width=True)
            
            st.write("**Free Energy Contributions**")
            cols = st.columns(3)
            for i in range(min(N+1, 3)):
                name = f"f_h_{i+1}"
                if name in plot_files:
                    with cols[i]:
                        st.plotly_chart(plot_files[name][0], use_container_width=True)
            if N >= 3:
                cols = st.columns(3)
                for i in range(3, min(N+1, 5)):
                    name = f"f_h_{i+1}"
                    if name in plot_files:
                        with cols[i-3]:
                            st.plotly_chart(plot_files[name][0], use_container_width=True)
            
            st.write("**Free Energy Density vs. c**")
            st.plotly_chart(plot_files["free_energy_vs_c"][0], use_container_width=True)
            
            try:
                with open(vts_file, "rb") as f:
                    st.download_button(
                        label=f"Download VTS (t={t:.3f} s)",
                        data=f,
                        file_name=vts_file.name,
                        mime="application/xml"
                    )
            except FileNotFoundError:
                st.warning(f"VTS file {vts_file.name} not found.")
            for name, (_, plot_file) in plot_files.items():
                try:
                    with open(plot_file, "rb") as f:
                        st.download_button(
                            label=f"Download {name} Plot (t={t:.3f} s)",
                            data=f,
                            file_name=plot_file.name,
                            mime="image/png"
                        )
                except FileNotFoundError:
                    st.warning(f"Plot file {plot_file.name} not found.")

        st.header("Total Free Energy Over Time")
        times, F_values = zip(*free_energies)
        fig_total_energy = px.line(x=times, y=np.array(F_values) * 1e6,  # Convert J to µJ
                                   labels={"x": "Time (s)", "y": "Total Free Energy (µJ)"},
                                   title="Total Free Energy Evolution")
        st.plotly_chart(fig_total_energy, use_container_width=True)

# Cleanup
if st.button("Clear Output Files"):
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=True)
    st.success("Output files cleared!")

# Instructions
st.markdown("""
### Instructions
1. Adjust parameters in the sidebar (e.g., nx, ny = 50–150 (50–150 µm), dx = 0.5–2 µm, dt = 0.00001–0.001 s).
2. Use Mobility Exponent = -6 (α = 10⁻⁶) and B = 2.0 MJ/m³ with free energy factor 5 for higher barrier height.
3. Keep κ_c = 0.5 J/m for strong interface energy and M_b = 10⁻²² m⁵/(J·s), M_s = 10⁻²¹ m⁵/(J·s) for stability.
4. Choose number of particles (N) and define centers (fraction of domain) and radii (5–20 µm).
5. View the initial density field (c) preview, with c = 1 in particles and c = 0 in vapor.
6. Click 'Completed Geometry' to confirm geometry.
7. Click 'Run Simulation' to start.
8. View Plotly heatmaps for c, η_1 (vapor), η_2 to η_{N+1} (grains), c·h_i², η_i², and f·h_i every 1 s.
9. Check free energy density vs. c plot (in MJ/m³).
10. Verify total free energy (µJ) decreases over time.
11. Download .vts files for ParaView and PNG plots.
12. Use 'Clear Output Files' to remove temporary files.

### Notes
- Free energy factor 5 and B = 2.0 MJ/m³ raise the powder phase barrier to ~10 MJ/m³.
- Mobility exponent = -6 (α = 10⁻⁶) ensures slow diffusion.
- Interface width δ = 5 µm ensures numerical stability with dx = 1 µm.
- Particle centers are normalized (0 to 1) relative to domain size (x_max = nx × dx µm).
""")
