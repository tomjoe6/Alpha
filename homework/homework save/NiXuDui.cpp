#include<iostream>
using namespace std;
int main()
{
    int n, shu0 = 0, shu1 = 0, shu2 = 0, jishu = 0;
    cin >> n;  
    char *p = new char[n];
    for(int i = 0; i < n; i++) {
        cin >> p[i];
    }
      
    for(int i = 0; i < n; i++) {
        if(p[i] == '0') {
            shu0++;
        }
        else if(p[i] == '1') {
            shu1++;
        }
        else if(p[i] == '2') {
            shu2++;
        }
    }
    
  
         for(int i = 0; i < shu0; i++) {
        if(p[i] != '0') {
            jishu++;
        }
    }
    
    for(int i = shu0; i < shu0 + shu1; i++) {
        if(p[i] == '2') {
            jishu++;
        }
    }
    
    
    delete []p;
    cout << jishu;
    return 0;
}
