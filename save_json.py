import os
import json


folder = "/home/zubairirshad/zero123/objaverse-rendering/partnet_mobility/"

output_folder = "/home/zubairirshad/zero123/objaverse-rendering"
# Get a list of all folders in the current directory
# folders = [folder for folder in os.listdir() if os.path.isdir(folder)]

all_folders = os.listdir(folder)

# # Iterate through each folder
# for folder in folders:
#     folder_path = os.path.join(os.getcwd(), folder)
    
#     # Get a list of all files in the folder
#     files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
    
#     # Create a dictionary with folder name as key and list of files as value
#     folder_data = {folder: files}
    
# Save the data as a JSON file
json_filename = os.path.join(output_folder, "valid_paths.json")
# json_filename = f"{folder}_data.json"
with open(json_filename, "w") as json_file:
    json.dump(all_folders, json_file, indent=2)

print(f"Saved data from {folder} to {json_filename}")