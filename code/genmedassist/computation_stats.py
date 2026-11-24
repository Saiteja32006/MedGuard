import json
import numpy as np
from pathlib import Path

# Define the path to your trajectories file
file_path = Path('system_output/trajectories/genmedassist_trajectories_v1.jsonl')

print("\n" + "="*50)
print(" INFERENCE TIME PERFORMANCE")
print("="*50)

if not file_path.exists():
    print(f"Error: Could not find '{file_path}'")
    print("Please check that the file and directory exist.")
else:
    # Load the data
    with open(file_path, encoding='utf-8') as f:
        trajs = [json.loads(line) for line in f]
    
    # Safely extract times (ignore missing entries instead of counting them as 0 ms)
    times = [t['processing_ms'] for t in trajs if 'processing_ms' in t]
    
    if not times:
        print("No valid 'processing_ms' data found in the file.")
    else:
        # Calculate statistics
        mean_time = np.mean(times)
        median_time = np.median(times)
        p95_time = np.percentile(times, 95)
        total_sec = sum(times) / 1000
        
        print("GenMedAssist Inference Time Per Case:")
        print(f"  • Valid Cases: {len(times)} (out of {len(trajs)} total lines)")
        print(f"  • Mean:        {mean_time:.1f} ms")
        print(f"  • Median:      {median_time:.1f} ms")
        print(f"  • P95:         {p95_time:.1f} ms")
        print(f"\n  • Total Time:  {total_sec:.1f} seconds")
        
        print("\nHardware & Setup:")
        print("  • Hardware: Intel i7, RTX A5000 24GB, 128GB RAM")
        print("  • GPU:      No GPU used for C-score (CPU only)")
        print("  • Training: No training required — inference-time method")

print("="*50 + "\n")