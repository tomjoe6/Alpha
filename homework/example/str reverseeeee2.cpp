#include <iostream>
#include <cstring>

void reverseString(char* str) {
    int jishu=0;
    for (int i=0;i<=99;i++){
    if(str[i]!=0){
        jishu++;
        
    }
    }
    for(int i=0;i<=(jishu/2);i++){
        char temp=str[jishu-i-1];
        str[jishu-i-1]=str[i];
        str[i]=temp;
    }

}

int main() {
    const int MAX_LENGTH = 100;
    char str[MAX_LENGTH]={0};

    std::cin.getline(str, MAX_LENGTH);

    reverseString(str);

    std::cout << "逆序后的字符串：" << str << std::endl;

    return 0;
}