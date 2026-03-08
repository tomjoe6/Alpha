#include<iostream>
using namespace std;


int *solve(int *s, int n, int m) {
    // 创建一个新数组来存储结果
    int p[110]={0};
for(int i=0;i<n;i++){
    if(i+m<n){
    p[i+m]=s[i];

    }
    else{
        p[i+m-n]=s[i];

    }

}
    return p;
    delete []p;

    /*********End**********/
}
int main()
{
    int n, m, s[110]={0};
    cin >> n >> m;

    for(int i = 0; i < n; i++)
        cin >> s[i];

    int *ans;
    /*********Begin*********/
    
    ans=solve(ans,n,m);
    /*********End**********/
    for(int i = 0; i < n; i++){
        if( i == 0){
            cout << *ans++;
        } else {
            cout << ' ' << *ans++;
        }
    }
    return 0;
}