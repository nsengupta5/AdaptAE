import matplotlib.pyplot as plt

"""
Create plots for the loss vs epoch, time vs epoch and loss vs time
"""
def create_plots(dataset, losses, times):
    # Plot the loss vs epoch
    plt.plot(losses)
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"plots/{dataset}-loss-vs-epoch.png")

    plt.clf()

    # Plot the time vs epoch
    plt.plot(times)
    plt.title("Time vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.savefig(f"plots/{dataset}-time-vs-epoch.png")

    plt.clf()

    # Plot the loss vs time
    plt.plot(losses, times)
    plt.title("Loss vs Time")
    plt.xlabel("Loss")
    plt.ylabel("Time (s)")
    plt.savefig(f"plots/{dataset}-loss-vs-time.png")
