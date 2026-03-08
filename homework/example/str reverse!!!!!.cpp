#include <iostream>
#include <cstring>
using namespace std;

int jilu1=0;
int i=0;
// 函数：翻转每个单词
void reverseWord(char str[]) {
    for(i=0;i<=9999;i++){
        if(str[i]==' '){continue;}
        else if(str[i]!=' '){
            for(int m=i+1;m<=9999;m++){
                jilu1=m;
                if(str[m]==' '||str[m]==0){
                    char temp;
                    int i1=i;
                    while(i1<=m-1){
                        temp=str[i1];
                        str[i1]=str[m-1];
                        str[m-1]=temp;
                        i1++;
                        m--;
                    }
                    
                    
                    i=jilu1;
                    break;}
                    
                
            }
        }
    }
}

int main() {
    char str[10000];
    // 从标准输入读取字符串
    cin.getline(str, 10000);

    // 翻转每个单词
    reverseWord(str);

    // 输出结果
    cout << str;

    return 0;



}