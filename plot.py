import matplotlib.pyplot as plt

def read_timings(filename):
    with open(filename, 'r') as f:
        timings_ms = [float(line.strip()) for line in f if line.strip()]
    return timings_ms

naive_ms = read_timings('timings_naive.txt')
lod_ms = read_timings('timings_lod.txt')

frames = range(1, len(naive_ms) + 1)

avg_naive = sum(naive_ms) / len(naive_ms)
avg_lod = sum(lod_ms) / len(lod_ms)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(frames, naive_ms, label=f'Naive Time (ms): avg {avg_naive:.2f}ms', color='blue')
plt.plot(frames, lod_ms, label=f'LOD Time (ms): avg {avg_lod:.2f}ms', color='orange')
plt.xlabel('Frame')
plt.ylabel('Time per Frame (ms)')
plt.title('Frame Time per Frame: Naive vs LOD')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.boxplot([naive_ms, lod_ms], tick_labels=['Naive', 'LOD'])
plt.ylabel('Time per Frame (ms)')
plt.title('Timing Distribution Comparison')

plt.tight_layout()
plt.show()
