import os
import math

from tqdm import tqdm
# --- Configuration ---
# Source directory containing the 'scene_xxxxx' folders.
source_dir = "/mnt/data/qingwen/Kubric_new/train"

# Destination directories where the symbolic links will be created.
dest_dir_part1 = "/mnt/data/qingwen/Kubric_new/part-static"
dest_dir_part2 = "/mnt/data/qingwen/Kubric_new/part-dynamic"

# The ratio for splitting the data into part1. 
# For example, 0.5 means 50% of the data will go to part1, and the rest to part2.
split_ratio = 0.5
# --- End of Configuration ---

def create_symlinks():
    """
    Finds all 'scene_*' directories in the source directory, splits them
    into two parts, and creates symbolic links in the destination directories.
    """
    try:
        # Get a sorted list of all entries in the source directory that start with 'scene_'
        all_scenes = sorted([d for d in os.listdir(source_dir) if d.startswith('scene_')])
        
        if not all_scenes:
            print(f"No 'scene_*' directories found in '{source_dir}'.")
            return

        # Calculate the split index based on the ratio
        split_index = math.floor(len(all_scenes) * split_ratio)

        # Split the list of scenes into two parts
        scenes_part1 = all_scenes[:split_index]
        scenes_part2 = all_scenes[split_index:]

        print(f"Total scenes found: {len(all_scenes)}")
        print(f"Splitting into {len(scenes_part1)} for part1 and {len(scenes_part2)} for part2.")

        # Create symbolic links for the first part
        print(f"\nCreating symbolic links in '{dest_dir_part1}'...")
        for scene_name in tqdm(scenes_part1):
            source_path = os.path.join(source_dir, scene_name)
            link_path = os.path.join(dest_dir_part1, scene_name)
            
            # Check if the link already exists
            if not os.path.lexists(link_path):
                os.symlink(source_path, link_path)
                # print(f"  Created link: {link_path} -> {source_path}")
            else:
                print(f"  Link already exists: {link_path}")

        # Create symbolic links for the second part
        print(f"\nCreating symbolic links in '{dest_dir_part2}'...")
        for scene_name in tqdm(scenes_part2):
            source_path = os.path.join(source_dir, scene_name)
            link_path = os.path.join(dest_dir_part2, scene_name)

            # Check if the link already exists
            if not os.path.lexists(link_path):
                os.symlink(source_path, link_path)
                # print(f"  Created link: {link_path} -> {source_path}")
            else:
                print(f"  Link already exists: {link_path}")
        
        print("\nScript finished successfully.")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please check if the source and destination directories are correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Ensure destination directories exist
    os.makedirs(dest_dir_part1, exist_ok=True)
    os.makedirs(dest_dir_part2, exist_ok=True)
    
    create_symlinks()