import os
import matplotlib.pyplot as plt
import scipy.io as spio
import numpy as np

# =============================================
# DATA LOADING AND PREPARATION
# =============================================

# Define the path and load data
data_path = r"C:\Users\user\Desktop\Internship_1\MATLAB_DMD\DATA\FLUIDS"
mat_files = [f for f in os.listdir(data_path) if f.endswith('.mat')]

if not mat_files:
    raise FileNotFoundError("No .mat files found in the directory.")

# Load the first .mat file
file_path = os.path.join(data_path, mat_files[0])
data = spio.loadmat(file_path)

# Extract grid dimensions and vorticity data
grid_height = data['m'][0][0]  # m
grid_width = data['n'][0][0]    # n
vorticity = data['VORTALL']     # Vorticity field

print(f"Grid dimensions: {grid_width}x{grid_height}")
print(f"Vorticity range: [{np.min(vorticity):.2f}, {np.max(vorticity):.2f}]")
print(f"Statistics: Mean = {np.mean(vorticity):.2f} ± {np.std(vorticity):.2f}")

# =============================================
# VISUALIZATION SETTINGS
# =============================================

# Color scaling (ignore extreme 1% outliers)
v_min, v_max = np.percentile(vorticity, [1, 99])
print(f"Color scale limits: [{v_min:.2f}, {v_max:.2f}]")

# Colormap (reversed RdBu for better contrast)
cmap = plt.cm.RdBu_r

# Manual contour levels (symmetric, focus on important ranges)
levels = [-17.0, -15.0, -12.0, -9.0, -6.0, -3.0, -1.0, -0.5, 
          0.5, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 17.0]

# =============================================
# PLOT CREATION
# =============================================

# Create figure with white background
fig, ax = plt.subplots(figsize=(11, 4), facecolor='white')

# Reshape vorticity data to 2D grid
vort_2d = np.reshape(vorticity[:, 0], (grid_width, grid_height)).T

# Filled contours (background)
filled = ax.contourf(
    vort_2d,
    levels=1001,        # High resolution for smooth gradients
    vmin=v_min,
    vmax=v_max,
    cmap=cmap,
    extend='both'       # Handle values beyond vmin/vmax
)

# Contour lines (foreground)
lines = ax.contour(
    vort_2d,
    levels=levels,      # Corrected variable name here
    colors='black',
    linewidths=0.8,
    linestyles='solid'
)

# Add labels to contour lines (optional)
ax.clabel(lines, inline=True, fontsize=8, fmt='%.1f')

# Add obstacle (circle)
obstacle = plt.Circle(
    (50, 100),          # Position (x,y)
    25,                 # Radius
    facecolor='white',
    edgecolor='black',
    linewidth=1.2,
    zorder=10           # Ensure obstacle is on top
)
ax.add_patch(obstacle)

# =============================================
# PLOT FORMATTING
# =============================================

# Remove axis elements
ax.set_xticks([])
ax.set_yticks([])
ax.set_aspect('equal')

# Hide plot borders
for spine in ax.spines.values():
    spine.set_visible(False)

# Ensure white background
ax.set_facecolor('white')

# =============================================
# DISPLAY OR SAVE
# =============================================

plt.tight_layout()
plt.show()




# ================================================
#   Performing POD to find out the  appropriate rank r (i.e., how many modes to keep in DMD).
# ================================================


X = vorticity # data matrix

X_mean = np.mean(X , axis =1 )
Y = X - X_mean[:,np.newaxis]


C = np.dot(Y.T , Y ) /(Y.shape[1] -1)

U,S ,V = np.linalg.svd(C)

Phi = np.dot(Y , U)
a = np.dot(Phi.T , Y )



print("Phi shape:", Phi.shape)
print("Each mode size (number of elements):", Phi.shape[0])
mode_shape = (449, 199)

fig, axes = plt.subplots(3, 3, figsize=(15, 14))  # 4 rows, 2 columns
axes = axes.flatten()  # Flatten to access axes easily

for i in range(9):
    mode = Phi[:, i].reshape(mode_shape)
    im = axes[i].imshow(mode.T, cmap='jet')  # Transpose for correct orientation
    axes[i].set_title(f'Mode {i+1}', fontsize=10)
    axes[i].axis('off')
    plt.colorbar(im, ax=axes[i], shrink=0.6)

# Hide the unused 8th subplot
axes[7].axis('off')

plt.tight_layout(pad=3.0)  # Increase padding between subplots
plt.show()



#===========================================
# Visualize the energy content of the modes
#===========================================


Energy = np.zeros((len(S) ,1 ))

for i in np.arange(0 , len(S)):
    Energy[i] = S[i]/np.sum(S)
    

X_Axis = np.arange(Energy.shape[0])
heights = Energy[:,0]


fig , axes = plt.subplots (1,2 , figsize=(12,5))
ax = axes[0]
ax.plot(Energy , marker ='o', markerfacecolor = 'none' , markeredgecolor = 'k' , ls ='-' , color = 'k')
ax.set_xlim(0 , 20)
ax.set_xlabel('Modes ')
ax.set_ylabel('Energy content')


ax = axes[1]
cumulative = np.cumsum(S)/np.sum(S)
ax.plot(cumulative, marker ='o' , markerfacecolor = 'none' , markeredgecolor = 'k' , ls = '-' , color = 'k')
ax.set_xlabel('Modes')
ax.set_ylabel('Cumulative energy ')
ax.set_xlim(0 , 20)
plt.show()


#========================================
#Performing DMD on the data
#========================================

def DMD (X1 , X2 ,r, dt):
    U, S , Vh = np.linalg.svd(X1 , full_matrices = False)
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vr = Vh.conj().T[:,:r]


    ### finding Atilde and find the eigenvalues and eigenvectorss
    Atilde = Ur.conj().T @ X2 @ Vr @ np.linalg.inv(Sr)
    Lambda, W = np.linalg.eig(Atilde)

    #DMD modes
    Phi = X2 @ Vr @ np.linalg.inv(Sr)@ W
    omega = np.log(Lambda)/dt

    alpha1 = np.linalg.lstsq(Phi , X1[:,0] , rcond = None)[0]
    b = np.linalg.lstsq(Phi , X2[:,0], rcond= None )[0]

    ## DMD reconstrcution
    time_dynamics = None
    for i in range(X1.shape[1]):
        v = np.array(alpha1)[:,0]*np.exp(np.array(omega)*(i+1)*dt)
        if time_dynamics is None:
            time_dynamics = v
        else :
            time_dynamics = np.vstack((time_dynamics, v))
    X_dmd = np.dot(np.array(Phi) , time_dynamics.T)
    return Phi , omega , Lambda , alpha1 , b , X_dmd , time_dynamics
    
X1 = np.matrix(X[: , 0:-1])
X2 = np.matrix(X[:, 1:])

r = 21
dt = 0.02*10

Phi, omega, Lambda, alpha1, b, X_dmd , time_dynamics = DMD(X1, X2, r, dt)

# Visualisation

plt.figure(figsize=(5,5))
plt.plot(np.real(Lambda), np.imag(Lambda), 'o', markersize=5)
plt.axhline(0, color='red', linestyle='--')
plt.axvline(0, color='red', linestyle='--')
plt.title('DMD Eigenvalues (Lambda)')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.grid(True)
plt.axis('equal')
plt.show()

for i in range(6):  # First 6 modes
    plt.plot(np.abs(time_dynamics[:, i]), label=f'Mode {i+1}')
plt.xlabel('Time step')
plt.ylabel('Amplitude')
plt.title('Time Dynamics of DMD Modes')
plt.legend()
plt.grid(True)
plt.show()

frame_index = 50  # Choose any index

original = X1[:, frame_index].reshape((449, 199))
reconstructed = X_dmd[:, frame_index].reshape((449, 199))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(original.T, cmap='seismic', origin='lower', vmin=-2.79, vmax=2.79)
plt.title('Original Frame')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(np.real(reconstructed).T, cmap='seismic', origin='lower', vmin=-2.79, vmax=2.79)
plt.title('Reconstructed Frame (DMD)')
plt.colorbar()

plt.tight_layout()
plt.show()











