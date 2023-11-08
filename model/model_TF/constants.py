import json
import tensorflow as tf

log_dir = "model_05"

with open(f"{log_dir}/setting.json", "r") as setting_file:
	setting = json.load(setting_file)

