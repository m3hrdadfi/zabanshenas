import os
import shutil
import glob

checkpoint_dir = "../models"
checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint-*"))
checkpoints = sorted(checkpoints, key=lambda n: int(n.split("-")[-1]))
if len(checkpoints) >= 3:
    shutil.rmtree(checkpoints[0])