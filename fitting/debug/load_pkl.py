import pickle
import pprint
import sys
import numpy as np

def load_nlf_pkl(file_path):
    """
    Load and visualize the contents of an NLF .pkl file, showing the type of 'allFrameHumans' but skipping its detailed inspection.

    :param file_path: Path to the .pkl file.
    """
    try:
        with open(file_path, 'rb') as file:
            dataPKL = pickle.load(file)
            print("\n[INFO] Successfully loaded the .pkl file.")
            
            # Display the keys in the .pkl file
            print("\n[INFO] Keys in the .pkl file:")
            pprint.pprint(dataPKL.keys())

            # Iterate through each key and display its shape and values
            print("\n[INFO] Attributes and their shapes/values:")
            for key, value in dataPKL.items():
                print("-" * 40)  # Separator for clarity
                if key == 'allFrameHumans':
                    print(f"Key: {key}\nType: {type(value)}\nNote: Skipped detailed inspection (use a different script to inspect this key)")
                    continue
                
                if isinstance(value, np.ndarray):
                    print(f"Key: {key}\nType: NumPy Array\nShape: {value.shape}\nValues:\n{value}")
                elif isinstance(value, list):
                    print(f"Key: {key}\nType: List\nLength: {len(value)}\nValues:\n{value}")
                elif isinstance(value, (int, float, str)):
                    print(f"Key: {key}\nType: Scalar ({type(value).__name__})\nValue: {value}")
                elif isinstance(value, dict):
                    print(f"Key: {key}\nType: Dictionary\nKeys: {list(value.keys())}")
                else:
                    print(f"Key: {key}\nType: {type(value)}\nValue: {value}")

    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
    except pickle.UnpicklingError:
        print(f"[ERROR] Failed to load the .pkl file. It may be corrupted or not a valid pickle file.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python loadNLFpkl.py <path_to_nlf_pkl>")
        sys.exit(1)

    pkl_file_path = sys.argv[1]
    load_nlf_pkl(pkl_file_path)