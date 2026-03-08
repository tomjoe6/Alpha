#include <iostream>
#include <string>
#include <cmath>
using namespace std;

int main(){
    int x=0;
    char a[15]={0};
 cin>>a;
 int n=0;
 while (a[n]!='\0'){
 ++n;
}
 if (n==0) cout<<x<<endl;
 else {
 for (long long i=1;i<=pow(2,n);++i){
 int *p=new int [n]{0};
 string s; 
 int m=i,j=0;
 while (m!=0){
 p[j]=m%2;
 m=m/2;
 ++j;
 }
 for (int g=0;g<n;++g){
 if (p[g]==1) {s+=a[g];}
 }
 bool flag=true;
 int h=s.length();
for (int q=1;q<=h/2;++q){
 if (s[q-1]!=s[h-q]) flag=false;
 }
 if (flag) ++x;
 delete [] p;
 }
 cout<<x-1<<endl;}
 return 0;
}

#include <iostream>
#include <string>
using namespace std;

// 检查字符串是否为回文
bool isPalindrome(const string& s) {
    int len = s.length();
    for (int i = 0; i < len / 2; ++i) {
        if (s[i] != s[len - i - 1]) {
            return false;
        }
    }
    return true;
}
//下面是AI给的优化代码

// 递归生成所有子集并检查回文
void generateSubsets(const string& a, int index, string current, int& count) {
    if (index == a.length()) {
        if (!current.empty() && isPalindrome(current)) {
            ++count;
        }
        return;
    }
    // 不包含当前字符
    generateSubsets(a, index + 1, current, count);
    // 包含当前字符
    generateSubsets(a, index + 1, current + a[index], count);
}

int main() {
    char a[15] = {0};
    cin >> a;
    string input(a);
    int count = 0;
    generateSubsets(input, 0, "", count);
    cout << count << endl;
    return 0;
}