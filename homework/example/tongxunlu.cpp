#include<iostream>
 
using namespace std;

int main()
{
    struct str{
        
        int y;
        int m;
        int d;
        long long phone;
        char add[51];
        char name[21];
    };
    
    int n;
    cin>>n;
    str *student=new str [n];
    
    for(int i=0;i<n;i++){
        
        cin>>student[i].name>>student[i].y>>student[i].m>>student[i].d>>student[i].phone;
        cin.ignore();
        cin.getline(student[i].add,50);
    }
    for(int m=0;m<=50;m++)
    {
        for(int o=0;o<=12;o++){
            for(int p=0;p<=31;p++){
                for(int i=0;i<n;i++){
                    if((2025-student[i].y)==m&&(12-student[i].m==o&&(31-student[i].d==p))){

                cout<<student[i].name<<" "<<student[i].y<<" "<<student[i].m<<" "<<student[i].d<<" "<<student[i].phone<<" "<<student[i].add<<endl;      
                }
                }
            
        }
        
        }
    }
    delete[]student;
   

    
    
    
       
}
       
