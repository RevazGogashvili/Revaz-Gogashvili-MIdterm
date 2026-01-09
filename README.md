## Task 1: Pearson Correlation Coefficient Analysis

### 1. Introduction
The objective of this task was to analyze the linear relationship between two variables, $X$ and $Y$, using a specific set of 8 data points. The primary metric used for this evaluation is the Pearson Correlation Coefficient ($r$), which measures the strength and direction of the linear relationship between two variables.

### 2. Dataset
The following data points were extracted and utilized for the calculation:

| Point ($i$) | $x_i$ | $y_i$ |
| :--- | :--- | :--- |
| 1 | -6.95 | 1.19 |
| 2 | -5.04 | 2.03 |
| 3 | -3.01 | -1.05 |
| 4 | -1.04 | 1.07 |
| 5 | 1.02 | -2.05 |
| 6 | 3.02 | 1.07 |
| 7 | 5.02 | -3.09 |
| 8 | 7.02 | -2.09 |

### 3. Methodology
The calculation follows the standard Pearson formula:

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

#### Step-by-Step Execution:
1.  **Mean Calculation:** The arithmetic means for both datasets were calculated.
    *   $\bar{x} = 0.005$
    *   $\bar{y} = -0.365$
2.  **Deviation Scores:** For each point, the deviation from the mean was determined ($x_i - \bar{x}$) and ($y_i - \bar{y}$).
3.  **Summation:**
    *   Sum of the products of deviations (Numerator): **-45.483**
    *   Sum of squared deviations for $X$: **168.487**
    *   Sum of squared deviations for $Y$: **25.982**
4.  **Final Coefficient:** The ratio was calculated by dividing the numerator by the square root of the product of the squared deviations.

### 4. Results & Visualization
The calculated Pearson Correlation Coefficient is:
**$r \approx -0.6874$**

#### Visualization
![Correlation Plot](./correlation_plot.png)
*Note: The plot uses a standardized scale of -10 to 10 on both axes to provide a clear view of the data distribution and the resulting negative trend line.*

### 5. Interpretation
The result of $r \approx -0.69$ indicates a **moderate-to-strong negative linear correlation**. 
- **Direction:** The negative sign indicates that as $X$ increases, $Y$ generally tends to decrease.
- **Strength:** Since $|r|$ is greater than 0.6 but less than 0.8, the relationship is considered substantial but contains some variance (noise), as evidenced by the "zig-zag" nature of the blue data points in the plot.
