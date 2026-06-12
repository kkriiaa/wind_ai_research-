import numpy as np
import matplotlib.pyplot as plt

# --- 1. PAPER PARAMETERS (CASE 2 DOMAIN + CASE 1 MPPT) ---
DOMAIN = 1500.0      # 1500m x 1500m (From Paper Page 9943)
N_TURBINES = 16      # 16 Turbines (4x4 Grid)
D = 126.0            # NREL 5MW Rotor Diameter
V_IN = 11.4          # Rated Wind Speed for NREL 5MW
RHO = 1.225          
K_WAKE = 0.0384      # Calculated offshore decay constant

# Generate 4x4 Grid Layout (Baseline)
grid_x = np.linspace(200, 1300, 4)
grid_y = np.linspace(200, 1300, 4)
xv, yv = np.meshgrid(grid_x, grid_y)
LAYOUT_X = xv.flatten()
LAYOUT_Y = yv.flatten()

# --- 2. PHYSICS: EMPIRICAL Cp & JENSEN WAKE ---
def calculate_cp(lam, beta):
    lam = np.clip(lam, 0.1, 20.0) 
    beta = np.clip(beta, 0.0, 30.0)
    lam_i_inv = (1.0 / (lam + 0.08 * beta)) - (0.035 / (beta**3 + 1.0))
    cp = 0.5176 * (116.0 * lam_i_inv - 0.4 * beta - 5.0) * np.exp(-21.0 * lam_i_inv) + 0.0068 * lam
    return max(0.0, cp)

def get_farm_performance(states):
    """ states: [lam1, beta1, lam2, beta2 ... lam16, beta16] """
    states = states.reshape(N_TURBINES, 2)
    v_local = np.full(N_TURBINES, V_IN)
    powers = np.zeros(N_TURBINES)
    
    # Sort by X to calculate wakes downstream
    idx = np.argsort(LAYOUT_X)
    
    for i in idx:
        lam, beta = states[i]
        cp = calculate_cp(lam, beta)
        ct = min(0.9, cp + 0.35)
        
        powers[i] = 0.5 * RHO * (np.pi * (D/2)**2) * (v_local[i]**3) * cp
        
        # Calculate wake for all turbines downstream of 'i'
        for j in idx:
            dx = LAYOUT_X[j] - LAYOUT_X[i]
            dy = abs(LAYOUT_Y[j] - LAYOUT_Y[i])
            if dx > 0.1:
                r_wake = (D/2) + K_WAKE * dx
                if dy < r_wake:
                    defic = (1 - np.sqrt(1 - ct)) / (1 + (K_WAKE * dx / (D/2)))**2
                    v_local[j] *= (1 - defic)
                    
    return np.sum(powers) / 1e6, v_local

# --- 3. ALGORITHM: HYBRID PSO-GA ---
def run_hybrid_optimization(iters=60):
    n_particles = 25
    dim = N_TURBINES * 2 # 32 dimensions
    
    # Init: Lambda [6-10], Beta [0-5]
    pos = np.random.uniform([6.0, 0.0]*N_TURBINES, [10.0, 2.0]*N_TURBINES, (n_particles, dim))
    vel = np.zeros_like(pos)
    pbest = pos.copy()
    pbest_fit = np.array([get_farm_performance(p)[0] for p in pos])
    gbest = pbest[np.argmax(pbest_fit)]
    gbest_fit = np.max(pbest_fit)
    
    history = []
    for t in range(iters):
        w = 0.8 - (t/iters)*0.4
        for i in range(n_particles):
            # PSO Update
            vel[i] = w*vel[i] + 1.5*np.random.rand()*(pbest[i]-pos[i]) + 1.5*np.random.rand()*(gbest-pos[i])
            
            # GA MUTATION: 15% chance to "jump" to prevent local optima stall
            if np.random.rand() < 0.15:
                vel[i] += np.random.normal(0, 1.5, dim)
                
            pos[i] += vel[i]
            # Clipping Lambda [2, 14] and Beta [0, 20]
            pos[i] = np.clip(pos[i], [2.0, 0.0]*N_TURBINES, [14.0, 20.0]*N_TURBINES)
            
            fit, _ = get_farm_performance(pos[i])
            if fit > pbest_fit[i]:
                pbest_fit[i], pbest[i] = fit, pos[i]
                if fit > gbest_fit:
                    gbest_fit, gbest = fit, pos[i]
        
        history.append(gbest_fit)
        print(f"Iteration {t+1}/{iters} | Farm Power: {gbest_fit:.2f} MW", end='\r')
        
    return gbest, history

# --- 4. EXECUTION ---
print(f"Optimizing MPPT for {N_TURBINES} Turbines (4x4 Grid)...")
best_params, history = run_hybrid_optimization()
final_p, final_winds = get_farm_performance(best_params)

# --- 5. VISUALIZATION ---
plt.figure(figsize=(15, 6))

# Plot 1: Convergence
plt.subplot(1, 2, 1)
plt.plot(history, color='blue', linewidth=2)
plt.title(f"Hybrid PSO-GA Convergence\n(16 Turbines, 1500m Domain)")
plt.xlabel("Iteration"); plt.ylabel("Total Power (MW)")

# Plot 2: Wake Visualization
plt.subplot(1, 2, 2)
X, Y = np.meshgrid(np.linspace(-100, 1600, 150), np.linspace(-100, 1600, 150))
U = np.full_like(X, V_IN)

params = best_params.reshape(N_TURBINES, 2)
for i in range(N_TURBINES):
    lam, beta = params[i]
    cp = calculate_cp(lam, beta)
    ct = min(0.9, cp + 0.35)
    dx = np.maximum(0.1, X - LAYOUT_X[i])
    dy = np.abs(Y - LAYOUT_Y[i])
    r_wake = (D/2) + K_WAKE * dx
    in_wake = (X > LAYOUT_X[i]) & (dy < r_wake)
    deficit = np.zeros_like(X)
    deficit[in_wake] = (1 - np.sqrt(1 - ct)) / (1 + (K_WAKE * dx[in_wake] / (D/2)))**2
    U *= (1 - deficit)

plt.contourf(X, Y, U, levels=20, cmap='RdYlBu_r')
plt.colorbar(label='Wind Speed (m/s)')
plt.scatter(LAYOUT_X, LAYOUT_Y, c='black', marker='x', label='Turbines')
plt.title(f"16-Turbine Optimized Wake Map\nTotal Power: {final_p:.2f} MW")
plt.legend()
plt.tight_layout()
plt.show()

print(f"\n\nOptimization Complete.")
print(f"Final Farm Power: {final_p:.2f} MW")
