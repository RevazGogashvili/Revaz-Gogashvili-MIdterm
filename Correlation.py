import numpy as np
import matplotlib.pyplot as plt

x = np.array([-6.95, -5.04, -3.01, -1.04, 1.02, 3.02, 5.02, 7.02])
y = np.array([1.19, 2.03, -1.05, 1.07, -2.05, 1.07, -3.09, -2.09])
n = len(x)

mean_x = np.sum(x) / n
mean_y = np.sum(y) / n

diff_x = x - mean_x
diff_y = y - mean_y

numerator = np.sum(diff_x * diff_y)
sum_sq_diff_x = np.sum(diff_x**2)
sum_sq_diff_y = np.sum(diff_y**2)
denominator = np.sqrt(sum_sq_diff_x * sum_sq_diff_y)

r = numerator / denominator

print("--- CALCULATION STEPS ---")
print(f"Number of points (n): {n}")
print(f"Mean of X: {mean_x:.4f}")
print(f"Mean of Y: {mean_y:.4f}")
print(f"Sum of products [Numerator]: {numerator:.4f}")
print(f"Sum of squared diff X: {sum_sq_diff_x:.4f}")
print(f"Sum of squared diff Y: {sum_sq_diff_y:.4f}")
print(f"Pearson Correlation Coefficient (r): {r:.4f}")
print("-------------------------\n")

plt.figure(figsize=(8, 8)) # Square figure to match the -10 to 10 scale

plt.xlim(-10, 10)
plt.ylim(-10, 10)

plt.scatter(x, y, color='dodgerblue', s=80, edgecolors='black', label='Data Points', zorder=3)

m, b = np.polyfit(x, y, 1)
line_x = np.array([-10, 10]) # Extend line to edges of plot
plt.plot(line_x, m*line_x + b, color='crimson', linestyle='--', linewidth=2,
         label=f'Linear Fit (r = {r:.3f})')

plt.title('Task 1: Correlation Analysis (-10 to 10 Scale)', fontsize=14)
plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)

plt.axhline(0, color='black', linewidth=1.5)
plt.axvline(0, color='black', linewidth=1.5)

plt.grid(True, which='both', linestyle=':', alpha=0.5)
plt.xticks(np.arange(-10, 11, 1))
plt.yticks(np.arange(-10, 11, 1))

plt.legend(loc='upper right')

plt.text(-9.5, 9, f'Pearson r = {r:.4f}', fontsize=12,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()