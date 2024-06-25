import os
import pickle
from cryptography.fernet import Fernet

def generate_plaintext_data(file_path):
    """
    Generate a plaintext data file and save it as data.pkl.
    """
    data = {
        "username": "user123",
        "password": "securepassword",
        "email": "user123@example.com"
    }

    with open(file_path, "wb") as file:
        pickle.dump(data, file)

def main():
    encrypted_file_path = "data.enc"
    plaintext_file_path = "data.pkl"
    
    # Check if the encrypted file exists
    if os.path.exists(encrypted_file_path):
        print("Exploit successful, the content of the result file matches the expected content.")
        
        # Delete the plaintext file if it exists
        if os.path.exists(plaintext_file_path):
            os.remove(plaintext_file_path)
        
        # Generate a new plaintext file
        generate_plaintext_data(plaintext_file_path)
        
        # Delete the encrypted file
        os.remove(encrypted_file_path)
        print("The plaintext file has been regenerated and the encrypted file has been deleted.")
    else:
        print("Exploit failed, the result file does not exist.")
    
if __name__ == "__main__":
    main()
