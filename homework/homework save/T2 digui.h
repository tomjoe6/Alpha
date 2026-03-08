#include<iostream>
using namespace std;
void deletechar(char *str1,const char*str2){
    if(str2[0]!=0){
        for(int i=0;str1[i]!=0;i++){
        if(str1[i]==str2[0]){
            for(int m=i;str1[m]!=0;m++){
                str1[m]=str1[m+1];
                }
        }
        }
    deletechar(str1,str2+1);
    }
}
         