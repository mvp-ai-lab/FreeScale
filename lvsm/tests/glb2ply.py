import trimesh
import os

def convert_glb_to_ply(input_path, output_path):
  """
  Converts a GLB file to a PLY file.

  Args:
    input_path (str): The file path of the input GLB file.
    output_path (str): The file path to save the output PLY file.
  """
  # Load the GLB file
  # The `force='mesh'` argument ensures that the result is a single mesh object.
  mesh = trimesh.load(input_path)

  # Export the mesh to a PLY file
  mesh.export(output_path, file_type='ply')

  print(f"Successfully converted {input_path} to {output_path}")

# --- Example Usage ---
# Replace with the actual path to your GLB file
input_file = "/home/qingwen/Pictures/glbscene_50_All_maskbFalse_maskwFalse_camFalse_skyFalse_predPointmap_Branch.glb" 

# Create an output filename
output_file = os.path.splitext(input_file)[0] + ".ply"

# Perform the conversion
if os.path.exists(input_file):
  convert_glb_to_ply(input_file, output_file)
else:
  print(f"Error: The file {input_file} was not found.")