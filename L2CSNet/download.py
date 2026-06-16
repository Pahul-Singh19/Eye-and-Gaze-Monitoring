from huggingface_hub import hf_hub_download
import shutil
import os

print('Downloading... please wait (95MB)')

path = hf_hub_download(
    repo_id='ashawkey/conversion',
    filename='L2CSNet_gaze360.pkl',
    repo_type='model'
)

dest = 'C:/Users/Harshita/Desktop/L2CSNet/models/L2CSNet_gaze360.pkl'
shutil.copy(path, dest)
print('Done! File saved to:', dest)
print('File size:', os.path.getsize(dest), 'bytes')
