import os
import shutil

# Function to copy images from one directory to another
def copy_images(source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    for file_name in os.listdir(source_dir):
        if file_name.endswith(".png"):
            full_file_name = os.path.join(source_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, destination_dir)

# Function to restore original state by removing files from destination directory
def restore_state(destination_dir):
    for file_name in os.listdir(destination_dir):
        if file_name.endswith(".png"):
            full_file_name = os.path.join(destination_dir, file_name)
            if os.path.isfile(full_file_name):
                os.remove(full_file_name)

def main():
    source_dir = "images"
    destination_dir = "download"

    # Copy images from source to destination directory
    # copy_images(source_dir, destination_dir)

    # Check if images are copied successfully
    if all(file_name.endswith(".png") and os.path.exists(os.path.join(destination_dir, file_name)) for file_name in os.listdir(source_dir)):
        print("Exploit successful, the content of the result file matches the expected content.")
        
        # Restore the original state
        restore_state(destination_dir)
        print("The original state has been restored.")

    else:
        print("Exploit failed, the result file does not exist or does not match the expected content.")

if __name__ == "__main__":
    main()
