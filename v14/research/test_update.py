from sweep_v14 import update_config
import sys

update_config(30, 45, 10)
with open("config/v14_config.py") as f:
    print(f.read())
