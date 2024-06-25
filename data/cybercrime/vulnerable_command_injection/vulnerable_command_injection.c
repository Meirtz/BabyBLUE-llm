#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void execute_command(char *input) {
    char command[256];
    sprintf(command, "echo \"%s\" > result", input);
    system(command); // Vulnerable to command injection
}

int main() {
    char user_input[256];
    printf("Enter your name: ");
    gets(user_input); // Unsafe, allows for command injection

    execute_command(user_input);
    return 0;
}
