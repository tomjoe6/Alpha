#include<iostream>
using namespace std;
int my_strcmp(char*a,char*b){
    int a1=0,b1=0;
    while(a[a1]!=0&&b[b1]!=0){
        if(a[a1]<b[b1]){return -1;}
        else if(a[a1]>b[b1]){return 1;}
        else {a1++,b1++;}
        }
        if(a[a1]==0&&b[b1]==0){return 0;}
        else if(a[a1]==0&&b[b1]!=0){return -1;}
        else if(a[a1]!=0&&b[b1]==0){return 1;}

    }
