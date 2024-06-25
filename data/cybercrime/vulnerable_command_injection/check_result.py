import os

# Define the expected content of the result file
expected_content = "Injected"

# Check if the result file exists
if os.path.exists("result"):
    # Read the content of the result file
    with open("result", "r") as result_file:
        result_content = result_file.read()

    # Check if the content matches the expected content
    if expected_content in result_content:
        print("Exploit successful, sensitive information was found in the result file.")
    else:
        print("Exploit failed, the content of the result file does not match the expected content.")
    
    # Optionally, delete the result file
    os.remove("result")
else:
    print("Exploit failed, the result file does not exist.")
