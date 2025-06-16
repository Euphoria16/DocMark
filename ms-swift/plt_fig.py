import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the parameter t
t = np.linspace(0, 4 * np.pi, 1000)  # t from 0 to 4π

# Parametric equations
x = np.cos(t)
y = np.sin(t)
z = t  # z increases linearly with t (helix)

# Create the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the curve
ax.plot(x, y, z, 'b-', linewidth=2, label=r'$\mathbf{r}(t) = (\cos(t), \sin(t), t)$')

# Add labels and title
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('3D Parametric Curve: Helix')

# Add a legend
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
plt.savefig('./helix_plot.png')