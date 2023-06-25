import numpy as np
import matplotlib.pyplot as plt

# mu : mean - sigma : stddev
def draw_gauss(mu, sigma):
    x = np.linspace(-10, 10, 10000)
    y = (1 / (np.sqrt((2 * np.pi)) * sigma)) * np.e ** (-0.5 * ((x - mu) / sigma) ** 2)
    
    plt.title('Gauss Eğrisi', fontweight='bold')
    plt.xlabel('x', fontweight='bold')
    plt.ylabel('Y', fontweight='bold')
    plt.plot(x, y)
    plt.show()
    
draw_gauss(0, 1)

# mu : mean - sigma : stddev
def draw_gauss_cartesian(mu, sigma):
    x = np.linspace(-10, 10, 10000)
    y = (1 / (np.sqrt((2 * np.pi)) * sigma)) * np.e ** (-0.5 * ((x - mu) / sigma) ** 2)
    
    plt.figure(figsize=(8, 8))
    plt.title('Gauss Eğrisi', fontweight='bold', pad=40, fontsize=14)
    plt.xlabel('x', fontweight='bold', position=(0.55, 0.8))
    plt.ylabel('Y', fontweight='bold', position=(0.55, 0.8))
    plt.ylim((-0.4, 0.4))
    
    axis = plt.gca()
    axis.spines['left'].set_position('center')
    axis.spines['bottom'].set_position('center')
    axis.spines['top'].set_color(None)
    axis.spines['right'].set_color(None)
    
    plt.plot(x, y)
    plt.show()
    
draw_gauss_cartesian(0, 1)
draw_gauss_cartesian(5, 1)
draw_gauss_cartesian(-5, 1)

draw_gauss_cartesian(0, 1)
draw_gauss_cartesian(0, 2)
draw_gauss_cartesian(0, 3)
