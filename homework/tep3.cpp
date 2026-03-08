#include <iostream>
#include <cstring>
using namespace std;
char * strrchr(char* str, char ch) 
{
    int ji;
    int n=strlen(str);
    for (int i=n;i>=0;i--){
        if(str[i]==ch){
            return str+i;


        }
    }

    return NULL;
}

int main()
{    
    char a[100]={0};
    char ch;
    cin.getline(a,100);
    cin>>ch;
    if(strrchr(a,ch)==NULL){
        cout<<"no";

    }
    else {
        cout<<strrchr(a,ch);
    }

    return 0;
}