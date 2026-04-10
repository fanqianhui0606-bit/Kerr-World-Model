import h5py
import numpy as np
import json
import os

def generate_dummy_data():
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Generate tiny dummy training data
    # Shape: (samples, time_steps, grid_points, channels)
    # The benchmark engine expects index 42 and t_start 100, so we need at least 43 samples and 201 steps
    print("生成测试用的虚拟数据 (Dummy Data)...")
    file_path = os.path.join(data_dir, 'training_data.h5')
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('u', (50, 250, 600, 6), dtype='f4', data=np.random.normal(0, 0.01, (50, 250, 600, 6)))
    
    # 2. Generate normalization stats
    stats = {
        "u": {
            "means": [0.0] * 6,
            "stds": [1.0] * 6
        }
    }
    with open(os.path.join(data_dir, 'norm_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
        
    print(f"✅ 虚拟数据生成完成: {file_path}")
    print("现在可以运行 scripts/benchmark_engine.py 进行演示。")

if __name__ == "__main__":
    generate_dummy_data()
