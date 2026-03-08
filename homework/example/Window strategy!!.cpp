#include <iostream>
using namespace std;
int main()
{
    char a[100]={0};
    cin.getline(a,100);
    int zuo=0,you=1,max=1,panduan=0;
    while(a[you]!=0){ 
        if(max<(you-zuo)){max=you-zuo;}
        else if(zuo==you){you++;}
    for(int i=you-1;i>=zuo;i--){
        if(a[you]==a[i]){zuo++;break;}
        else if(i==zuo){you++;}
        
        }
        


        
     
    
    }
    cout<<max;
    return 0;
}