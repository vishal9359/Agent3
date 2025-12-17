// Simple calculator example for testing Agent5
#include <iostream>
#include <string>

bool isValidNumber(int n) {
    return n >= 0 && n <= 1000;
}

int parseNumber(const std::string& str) {
    try {
        return std::stoi(str);
    } catch (...) {
        return -1;
    }
}

int calculate(int a, int b, char op) {
    if (op == '+') {
        return a + b;
    } else if (op == '-') {
        return a - b;
    } else if (op == '*') {
        return a * b;
    } else if (op == '/') {
        if (b == 0) {
            return -1;
        }
        return a / b;
    }
    return -1;
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: calculator <num1> <op> <num2>" << std::endl;
        return 1;
    }
    
    int a = parseNumber(argv[1]);
    int b = parseNumber(argv[3]);
    char op = argv[2][0];
    
    if (!isValidNumber(a) || !isValidNumber(b)) {
        std::cerr << "Invalid numbers" << std::endl;
        return 1;
    }
    
    if (op != '+' && op != '-' && op != '*' && op != '/') {
        std::cerr << "Invalid operator" << std::endl;
        return 1;
    }
    
    int result = calculate(a, b, op);
    
    if (result == -1) {
        std::cerr << "Calculation error" << std::endl;
        return 1;
    }
    
    std::cout << "Result: " << result << std::endl;
    return 0;
}

