import pickle
import sys
import numpy as np

def load_pkl(file_path):
    """
    Load and visualize the contents of a .pkl file, focusing on the 'allFrameHumans' key.

    :param file_path: Path to the .pkl file.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print("\n[INFO] Successfully loaded the .pkl file.")
            
            # Check if 'allFrameHumans' exists
            if isinstance(data, dict) and 'allFrameHumans' in data:
                print("\n[INFO] Inspecting 'allFrameHumans'...")
                all_frame_humans = data['allFrameHumans']
                if len(all_frame_humans) > 0:
                    print(f"[INFO] Total frames: {len(all_frame_humans)}")
                    print(f"[INFO] Humans detected in the first frame: {len(all_frame_humans[0])}")
                    
                    # Get the first human's data
                    first_human = all_frame_humans[0][0]
                    
                    # Print the order of keys and their shapes/types
                    print("\n[INFO] Keys and their shapes/types for the first human:")
                    for key, value in first_human.items():
                        print("-" * 40)  # Separator for better clarity
                        if isinstance(value, np.ndarray):
                            print(f"Key: {key}\nType: NumPy Array\nShape: {value.shape}")
                        elif isinstance(value, list):
                            print(f"Key: {key}\nType: List\nLength: {len(value)}")
                        elif isinstance(value, (int, float, str)):
                            print(f"Key: {key}\nType: Scalar ({type(value).__name__})\nValue: {value}")
                        else:
                            print(f"Key: {key}\nType: {type(value)}")
                    
                    # Print the data for the first human
                    print("\n[INFO] Data for the first human in the first frame:")
                    print(first_human)
                else:
                    print("[INFO] No humans detected in the frames.")
            else:
                print("[INFO] 'allFrameHumans' key not found in the dictionary.")

    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
    except pickle.UnpicklingError:
        print(f"[ERROR] Failed to load the .pkl file. It may be corrupted or not a valid pickle file.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python loadPKL.py <path_to_pkl_file>")
        sys.exit(1)

    pkl_file_path = sys.argv[1]
    load_pkl(pkl_file_path)