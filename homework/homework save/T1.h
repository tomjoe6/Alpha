#include<iostream>
#include<cstring>
using namespace std;
void getDate(int &dd,int &mm,int &yy){
    int d,y;
    char m1[4];
    char m[12][4]={"Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"};
    cin>>d>>m1>>y;
    for(int i=0;i<=11;i++){
        if(strcmp(m[i],m1)==0){
            dd=d;mm=(i+1);yy=y;
        }
    }

}