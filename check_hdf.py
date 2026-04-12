import os
import h5py
import sys

data_dir = '/root/autodl-tmp/FY-4B/calibration/2000M/CH07'
files = sorted([f for f in os.listdir(data_dir) if f.endswith('.HDF')])

error_files = []
for f in files:
    filepath = os.path.join(data_dir, f)
    try:
        with h5py.File(filepath, 'r') as h:
            pass
    except Exception as e:
        error_files.append(f)
        print(f'{f}: ERROR - {e}')

print(f'\n总文件数: {len(files)}')
print(f'损坏文件数: {len(error_files)}')

if error_files:
    print('\n损坏文件列表:')
    for f in error_files:
        print(f'  {f}')
    sys.exit(1)
else:
    print('所有文件正常')
    sys.exit(0)
