import matplotlib.pyplot as plt
import numpy as np

# Set font family for the plot
plt.rcParams['font.family'] = 'Times New Roman'

# Example data
x = [1, 2, 1, 3, 1, 3, 2]
y = [1, 1, 2, 1, 3, 2, 3]
data = [4.64, 4.66, 4.65, 4.69, 4.66, 4.68, 4.69]  # Data values for each (x, y) pair

# Create figure and axis
fig, ax = plt.subplots()

# Scatter plot with data values as color
scatter = ax.scatter(x, y, c=data, cmap='viridis', s=100, edgecolor='black', alpha=0.7)

# Add colorbar to show the data values
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Data Value')

# Set labels for the x and y axes
# ax.set_xlabel(r'$\lambda_1$' + '\n Effect of ' + r'$\lambda_1$' + 'and' + r'$\lambda_2$', fontsize=12, labelpad=2)
ax.set_xlabel(r'$\lambda_1$', fontsize=18)
ax.set_ylabel(r'$\lambda_2$', fontsize=18)

# Add data labels near the points
for (i, j, d) in zip(x, y, data):
    ax.text(i, j, str(d), ha='center', va='bottom', fontsize=18)

# Add grid
ax.grid(alpha=0.3, linestyle='--')

# Save the plot as a PNG file
plt.savefig('scatter_plot.png', dpi=300)