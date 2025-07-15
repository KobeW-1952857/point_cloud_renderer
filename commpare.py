import matplotlib.pyplot as plt
import re
import numpy as np

def parse_timing_data(file_path):
    """
    Parses the timing data from the given file path.
    Extracts iteration numbers and their corresponding times.
    """
    times = []
    # Regex to match lines like "VOO 0 - 1.663ms" or "Naïve0 - 1.489ms"
    # It captures the number after the prefix (VOO or Naïve) and the time.
    # It also handles the "Naïve" character correctly.
    pattern = re.compile(r'(?:\S*)\s*(\d+)\s*-\s*([\d.]+)\s*ms')
    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    # We don't strictly need the iteration number for the plot,
                    # but it's good to parse it to ensure we're getting the right lines.
                    # iteration = int(match.group(1))
                    time = float(match.group(2))
                    times.append(time)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    return times

def print_statistics(name, times):
    if not times:
        print("No data to calculate statistics.")
        return
    
    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    q1, median, q3 = np.percentile(times, [25, 50, 75])
    print(f"--- {name} Timings Statistics ---")
    print(f"  Count:  {len(times)}")
    print(f"  Mean:   {avg_time:.3f}ms")
    print(f"  Median: {median:.3f}ms")
    print(f"  Min:    {min_time:.3f}ms")
    print(f"  Max:    {max_time:.3f}ms")
    print(f"  Q1:     {q1:.3f}ms")
    print(f"  Q3:     {q3:.3f}ms\n")



# Create a figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 7)) # Adjust figure size to accommodate two plots
datasets = ['Naive', 'VOO', 'VOO_early', 'VOO_dedup']
box_plot_data =[]

for dataset in datasets:
    times = parse_timing_data(f"build/{dataset}_timings.txt")
    print_statistics(dataset, times)
    avg_time = sum(times) / len(times) if times else 0
    # --- Line Plot (Left Subplot) ---
    axes[0].plot(times, label=f'{dataset} Timings (Avg: {avg_time:.2f}ms)', alpha=0.7)
    axes[0].axhline(y=avg_time, linestyle='--', label=f'{dataset} Average ({avg_time:.2f}ms)')
    # --- Boxplot (Right Subplot) ---
    box_plot_data.append(times)

axes[1].boxplot(box_plot_data, labels=datasets, patch_artist=True, medianprops={'color': 'black'})

# Add labels and title for the line plot
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Time (ms)')
axes[0].set_title(f'Comparison of {", ".join(datasets[:-1])} {datasets[-1]} Algorithm Timings per Iteration')
axes[0].legend()
axes[0].grid(True)

# Add labels and title for the boxplot
axes[1].set_xlabel('Algorithm')
axes[1].set_ylabel('Time (ms)')
axes[1].set_title('Distribution of VOO and Naïve Algorithm Timings')
axes[1].grid(True, axis='y')

plt.tight_layout() # Adjust layout to prevent overlapping titles/labels

plt.show()