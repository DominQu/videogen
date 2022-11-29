from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    file = Path("pennaction_ssim.txt")
    data = np.loadtxt(file)
    x = np.arange(data.shape[0])
    plt.scatter(x, data, s=10, c='green')
    plt.xlabel("Epoka treningowa")
    plt.ylabel("Wartość wsp. SSIM")
    # plt.ylim(bottom=0)
    plt.grid(True)
    # plt.show()
    plt.savefig("plots/pennaction_ssim_plot.png")