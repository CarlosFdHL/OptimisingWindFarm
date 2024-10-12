import matplotlib.pyplot as plt

# Lists to store the history of installed capacity and NPV
hist_MW = []
hist_npv = []

# Function to plot the stored results
def plot_results():
    plt.figure(figsize=(10, 6))

    # Plot installed capacity over iterations
    plt.subplot(2, 1, 1)
    plt.plot(hist_MW, label='Installed Capacity (MW)', marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Installed Capacity (MW)')
    plt.title('Installed Capacity over Iterations')
    plt.grid(True)
    plt.legend()

    # Plot NPV over iterations
    plt.subplot(2, 1, 2)
    plt.plot(hist_npv, label='NPV', marker='o', color='orange')
    plt.xlabel('Iteration')
    plt.ylabel('NPV')
    plt.title('NPV over Iterations')
    plt.grid(True)
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

