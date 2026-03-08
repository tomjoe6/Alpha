#include<iostream>
using namespace std;
void deletechar(char *str1,const char*str2){
    for(int i=0;str2[i]!=0;i++){
        for(int m=0;str1[m]!=0;m++){
            if(str1[m]==str2[i]){
                for(int n=m;str1[n]!=0;n++){
                    str1[n]=str1[n+1];
                }
            }
        }
    }
}