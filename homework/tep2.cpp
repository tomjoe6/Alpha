#include<iostream>
using namespace std;


char *itos(int n)
{   
    int weishu=0,a=n;
    while(a!=0){
        a=a/10;
        weishu++;}
    char*p=new char[2*weishu];
    int t=2*weishu-2;
    while(t>0){
        p[t]=n%10+'0';
        p[t-1]='-';
        t=t-2;
        n=n/10;

    }
    p[0]=n%10+'0';
    p[2*weishu-1]=0;
    return p;

}

int main()
{
    int n;
    cin >> n;
    char *str = itos(n);
    cout <<str;
    delete[]str;
    return 0;
}