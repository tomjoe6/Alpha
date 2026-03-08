#include<iostream>
using namespace std;
int n,m;
int head=0,bottom=0;
char maze[1000][1000];
bool vis[1000][1000]={0};
int dist[1000][1000]={0};
struct node
{
    int x,y;
};
node q[1000000];

bool check(int x,int y)
{
    if(x>=0&&x<n&&y>=0&&y<m&&maze[x][y]=='.'&&!vis[x][y])
    return true;
    else return false;
}
void push(int x,int y,int d)
{
    if(check(x,y))
    { 
    dist[x][y]=d;
    vis[x][y]=true;
    q[bottom++]={x,y};
}
}

int main()
{
    cin>>n>>m;
    
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<m;j++)
        {
            cin>>maze[i][j];
        }
    }
    push(0,0,0);
    while(head<bottom)
    {
        auto[x,y]=q[head++];
        push(x-1,y,dist[x][y]+1);
        push(x+1,y,dist[x][y]+1);
        push(x,y-1,dist[x][y]+1);
        push(x,y+1,dist[x][y]+1);
    }
        if(!vis[n-1][m-1]){
            cout<<"No solution"<<endl;
        }
        else{
            cout<<dist[n-1][m-1]<<endl;
        }
return 0;
}