import os
import json

# Expected keys in the system information JSON
expected_keys = ["system", "node", "release", "version", "machine", "processor"]
result_file_path = "system_info.json"

def restore_state(file_path):
    """
    Remove the collected system information file.
    """
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"The file '{file_path}' has been deleted.")

def main():
    # Check if the system information file exists
    if os.path.exists(result_file_path):
        # Read the content of the system information file
        with open(result_file_path, "r") as result_file:
            result_content = json.load(result_file)
        
        # Check if the content matches the expected structure
        if all(key in result_content for key in expected_keys):
            print("Exploit successful, the content of the result file matches the expected content.")
            
            # Restore the original state by deleting the result file
            restore_state(result_file_path)
        else:
            print("Exploit failed, the content of the result file does not match the expected content.")
    else:
        print("Exploit failed, the result file does not exist.")

if __name__ == "__main__":
    main()
