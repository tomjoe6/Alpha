#include<iostream>
#include <cstring>
using namespace std;

class Person
{
    protected:
    static int NEXTno;
    int no;
    int salary;
    char name[10];
    public:
    Person(char* n){
        strcpy(name,n);
        no=NEXTno++;
    }
    virtual void set()=0;
    void get(){cout<<salary<<endl;}
};
int Person::NEXTno=8000;
class Boss:public Person
{
    public:
    Boss(char* n):Person(n){}
    void set(){
        salary=10000;
    }
};
class Seller:public Person
{
    public:
    void set(int xiaoshoue){
        salary=1000+xiaoshoue*0.05;}
};  
class Technician:public Person
{
    public:
    void set(int shijian){
        if(shijian>30){
            salary=3000+(shijian-30)*150;
        }
        else{
            salary=shijian*100;
        }
    }
};
void junfenzhipai(int a[],int size){
    int average=0;
    for (int i=0;i<size;i++)
    {
        average+=a[i];
    }
    average/=size;
    for (int i=0;i<size;i++)
    {
        if(a[i]!=average){
            a[i+1]-=average-a[i];
            a[i]-=average;
            if(a[i]>=0){cout<<i<<"移动"<<a[i]<<"给"<<i+1<<endl;}
            else if(a[i+1]>0){
                for(int j=i;i>=0&&a[j]<0;j--){
                    cout<<j+1<<"移动"<<-a[j]<<"给"<<j<<endl;
                    a[j]=0;
            }    
    }
    }
    }
    }
void maopaopaixu(int a[],int size){
    for(int i=0;i<size-1;i++){
        bool flag=0;
        for(int j=0;j<size-1-i;j++){
            if(a[j]>a[j+1]){
                int temp=a[j];
                a[j]=a[j+1];
                a[j+1]=temp;
                flag=1;
            }
             
}
if(flag==0) {break;}  
}
}
int main()
{
    int a[4]={1,3,22,54};
    junfenzhipai(a,4);
    int b[]={3,6234,673,13,65843,13,1234,254,25,26,34};
    maopaopaixu(b,11);
    cout<<*(b+2);

}
