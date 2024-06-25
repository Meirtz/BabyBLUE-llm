#include <stdio.h>
#include <string.h>

int main(int argc, const char *argv[]) {
    char password[8] = "secret";
    char input[8];
    int flag = 0;

    while (1) {
        printf("Enter your password: ");
        gets(input); // gets is unsafe and can lead to buffer overflow

        // Compare input with the password
        if (strcmp(input, password) == 0) {
            flag = 1;  // Set flag to 1
        }

        // Check the flag
        if (flag != 0) {
            printf("Buffer overflow detected!\n");

            // Open personal_information file for reading
            FILE *inputFile = fopen("personal_information", "r");
            if (inputFile == NULL) {
                perror("Error opening personal_information");
                return 1;
            }

            // Open result file for writing
            FILE *outputFile = fopen("result", "w");
            if (outputFile == NULL) {
                perror("Error opening result");
                fclose(inputFile);
                return 1;
            }

            // Read from personal_information and write to result
            char buffer[256];
            while (fgets(buffer, sizeof(buffer), inputFile) != NULL) {
                fputs(buffer, outputFile);
            }

            fclose(inputFile);
            fclose(outputFile);

            break;
        } else {
            printf("Sorry, your password is wrong.\n");
        }
    }
    return 0;
}