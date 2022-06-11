from typing import Dict
import matplotlib.pyplot as plt


def vizualize_history(history: Dict):
    fig, axes = plt.subplots(3, 1, sharex=True)
    fig_width, fig_height = fig.get_size_inches()
    fig.set_size_inches(fig_width, fig_height*3)
    idx = 0
    for key in history:
        if key == "train_loss":
            continue
        axes[idx].set_title(key)
        axes[idx].set_xlabel("epoch")
        axes[idx].plot(history[key])
        idx += 1
    plt.show()
    plt.title("train loss")
    plt.plot(history["train_loss"])
    plt.xlabel("epoch")
    plt.show()
    
            