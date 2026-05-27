import matplotlib.pyplot as plt
import numpy as np

# Set font family for the plot
plt.rcParams['font.family'] = 'Times New Roman'

# Data
x_axis_data = [64, 128, 256, 512, 1024]
y_axis_data1 = [4.88, 4.76, 4.64, 4.61, 4.60]  # 10%
resource_consumption = [10.2, 21.3, 45.2, 119.5, 302.6]  # Example resource consumption data

# Create figure and primary y-axis
fig, ax1 = plt.subplots()

# Plot the first y-axis data (MRE)
line1, = ax1.plot(x_axis_data, y_axis_data1, marker='o', alpha=1, linewidth=1.5, label='MRE', color='green')

# Set labels for the primary y-axis
ax1.set_xlabel(r'$\mathcal{d}$', fontsize=18)
# ax1.set_ylabel('MRE', fontsize=18)
ax1.grid(alpha=0.3, linestyle='--')

# Add data labels for the first y-axis data near the center
for a, b in zip(x_axis_data, y_axis_data1):
    ax1.text(a, b, str(b), ha='center', va='bottom', fontsize=18)

# Create secondary y-axis
ax2 = ax1.twinx()

# Plot the second y-axis data (Time Consumption)
line2, = ax2.plot(x_axis_data, resource_consumption, marker='D', alpha=1, linewidth=1.5, label='Time Consumption(ms)', color='red')
# ax2.set_ylabel('Time Consumption(ms)', fontsize=18)

# Add data labels for the secondary y-axis near the center
for a, b in zip(x_axis_data, resource_consumption):
    ax2.text(a, b, str(b), ha='center', va='bottom', fontsize=18, color='red')

# Combine legends from both y-axes
lines = [line1, line2]
labels = [line.get_label() for line in lines]
fig.legend(lines, labels, loc='center', bbox_to_anchor=(0.5, 0.8), ncol=2)

# Save the plot as a PNG file
plt.savefig('feature_dimension.png', dpi=1200)
