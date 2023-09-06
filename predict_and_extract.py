import os
import torch
import toml
import argparse

from src.predict import predict
from src.extract import intersection

try:
    import paths as paths
except:
    import src.paths as paths

parser = argparse.ArgumentParser()
parser.add_argument("city", help="City to be predicted, must be the same as the name of the dataset")
parser.add_argument("type", help="Roof Typology, Green for Greenroof, Solar for PV Roof")
args = parser.parse_args()

config = toml.load(os.path.join(paths.PROJECT_DIR, 'config/predict-config.toml'))
    
city_name = args.city
target_type = args.type

tiles_dir = os.path.join(paths.PROJECT_DIR, "results", '02Images', city_name)
mask_dir = os.path.join(paths.PROJECT_DIR, "results", "03Masks", target_type, city_name)
tile_size =  config["img_size"]

# load checkpoints
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if target_type == "Solar":
    checkpoint_path = config["checkpoint_path"]
    checkpoint_name = config["solar_checkpoint"]
    chkpt = torch.load(os.path.join(paths.PROJECT_DIR, checkpoint_path, checkpoint_name), map_location=device)

elif target_type == "Green":
    checkpoint_path = config["checkpoint_path"]
    checkpoint_name = config["green_checkpoint"]
    chkpt = torch.load(os.path.join(paths.PROJECT_DIR, checkpoint_path, checkpoint_name), map_location=device)

predict(tiles_dir, mask_dir, tile_size, device, chkpt)

#intersection(target_type, city_name, mask_dir)
