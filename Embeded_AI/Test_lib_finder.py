import os
import platform

def find_arduino_library_folder():
    system = platform.system()

    if system == "Windows":
        base_path = os.path.expanduser("~/Documents/Arduino/libraries/")
    elif system == "Linux":
        base_path = os.path.expanduser("~/Arduino/libraries/")
    elif system == "Darwin":  # macOS
        base_path = os.path.expanduser("~/Documents/Arduino/libraries/")
    else:
        raise OSError("Unsupported operating system")

    # Ensure the path exists
    if os.path.exists(base_path):
        return base_path.replace("\\", "/")
    else:
        print("Arduino library folder not found. Please specify manually.")
        return None

library_path = find_arduino_library_folder()
if library_path:
    print(f"Arduino library folder found: {library_path}")