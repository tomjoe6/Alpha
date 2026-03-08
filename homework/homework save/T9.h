#include<iostream>
using namespace std;
//请自定义一个函数并让一个指针指向这个函数，我们默认此函数只有一个根

double formula(double (*des)(double),double zuo,double you){
    double x1=(zuo+you)/2;
    if(des(zuo)*des(x1)==0||des(you)*des(x1)==0){return x1;}
    else if(des(x1)>0.001){
        if((des(zuo)*des(x1)<0)){formula(des,zuo,x1);}
        else if((des(you)*des(x1)<0)){formula(des,x1,you);}
    }

    else{return x1;};
}
