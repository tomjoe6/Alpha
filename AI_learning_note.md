# 记录学习档案
## Python
与C++很不同的点：  
①不用分号，使用缩进  
②输出默认换行，使用,来不换行，如 `print('a'),`

语法重点：  
输出列表，会把[]也输出出来  
列表/字符串可以截取，for instance `print (str[n:m:o])` 表示的是从索引从n到m隔o个输出一位  
正则表达式，如`result=re.match(r'(\w+) is \$(\d+)\.(\d+)' , 'rice is $5.00')`括号表示分组，使用`result.group(n)`选出第n个括号里的东西，且`+`表示匹配一个或多个，`?`表示零个或一个，`*`表示零个或多个

## Machine Learning
Regression：回归————输入一个或多个参数后输出标量  
Classification：输入一个事物，从已知选项里面输出一个  
Structure：创造  

 对于已知量的乘叫权重 