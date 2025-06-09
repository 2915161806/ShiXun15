## 一、环境准备
```python
# 检查Python版本
python --version

# 创建并激活虚拟环境
python -m venv myenv
# Windows
myenv\Scripts\activate
# Linux/Mac
source myenv/bin/activate

# 安装第三方库
pip install requests

二、变量、变量类型、作用域
基本变量类型：int、float、str、bool、list、tuple、dict、set。
作用域：全局变量、局部变量，global和nonlocal关键字。
类型转换：如int()、str()。
代码示例：
# 变量类型
name = "Alice"  # str
age = 20        # int
grades = [90, 85, 88]  # list
info = {"name": "Alice", "age": 20}  # dict

# 类型转换
age_str = str(age)
number = int("123")

# 作用域
x = 10  # 全局变量
def my_function():
    y = 5  # 局部变量
    global x
    x += 1
    print(f"Inside function: x={x}, y={y}")

my_function()
print(f"Outside function: x={x}")

三、运算符及表达式
算术运算符：+、-、*、/、//、%、**。
比较运算符：==、!=、>、<、>=、<=。
逻辑运算符：and、or、not。
位运算符：&、|、^、<<、>>。
代码示例：
# 算术运算
a = 10
b = 3
print(a + b)  # 13
print(a // b)  # 3（整除）
print(a ** b)  # 1000（幂）

# 逻辑运算
x = True
y = False
print(x and y)  # False
print(x or y)   # True

# 比较运算
print(a > b)  # True

四、语句：条件、循环、异常
条件语句：if、elif、else。
循环语句：for、while、break、continue。
异常处理：try、except、finally。
代码示例：
# 条件语句
score = 85
if score >= 90:
    print("A")
elif score >= 60:
    print("Pass")
else:
    print("Fail")

# 循环语句
for i in range(5):
    if i == 3:
        continue
    print(i)

# 异常处理
try:
    num = int(input("Enter a number: "))
    print(100 / num)
except ZeroDivisionError:
    print("Cannot divide by zero!")
except ValueError:
    print("Invalid input!")
finally:
    print("Execution completed.")

五、函数：定义、参数、匿名函数、高阶函数
函数定义：def关键字，默认参数，可变参数（*args、**kwargs）。
匿名函数：lambda。
高阶函数：接受函数作为参数或返回函数。
代码示例：
# 函数定义
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

print(greet("Alice"))  # Hello, Alice!
print(greet("Bob", "Hi"))  # Hi, Bob!

# 可变参数
def sum_numbers(*args):
    return sum(args)
print(sum_numbers(1, 2, 3, 4))  # 10

# 匿名函数
double = lambda x: x * 2
print(double(5))  # 10

# 高阶函数
def apply_func(func, value):
    return func(value)
print(apply_func(lambda x: x ** 2, 4))  # 16

六、包和模块：定义模块、导入模块、使用模块、第三方模块
模块：import语句，from ... import ...。
创建模块：一个.py文件。
包：包含__init__.py的文件夹。
第三方模块：如requests、numpy。
代码示例：
# 创建模块 mymodule.py
# mymodule.py
def say_hello():
    return "Hello from module!"

# 主程序
import mymodule
print(mymodule.say_hello())

# 导入第三方模块
import requests
response = requests.get("https://api.github.com")
print(response.status_code)  # 200

# 包使用示例
from mypackage import mymodule

七、类和对象
类定义：class关键字，属性和方法。
继承、多态、封装。
实例化对象。
代码示例：
# 定义类
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"I am {self.name}, {self.age} years old."

# 继承
class GradStudent(Student):
    def __init__(self, name, age, major):
        super().__init__(name, age)
        self.major = major

    def introduce(self):
        return f"I am {self.name}, a {self.major} student."

# 使用
student = Student("Alice", 20)
grad = GradStudent("Bob", 22, "CS")
print(student.introduce())  # I am Alice, 20 years old.
print(grad.introduce())     # I am Bob, a CS student.

八、装饰器
本质：高阶函数，接受函数并返回新函数。
@语法：简化装饰器应用。
带参数的装饰器。
代码示例：
# 简单装饰器
def my_decorator(func):
    def wrapper():
        print("Before function")
        func()
        print("After function")
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")

say_hello()

# 带参数的装饰器
def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hi, {name}!")

greet("Alice")

九、文件操作
文本文件读写：open()、read()、write()。
上下文管理器：with语句。
CSV、JSON 文件处理。
代码示例：
# 写文件
with open("example.txt", "w") as f:
    f.write("Hello, Python!\n")

# 读文件
with open("example.txt", "r") as f:
    content = f.read()
    print(content)

# 处理CSV
import csv
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Name", "Age"])
    writer.writerow(["Alice", 20])

十、Git 命令操作
git init：初始化仓库
git add .：添加到暂存区
git commit -m "message"：提交更改
git remote add origin "url"：添加远程仓库
git pull --rebase origin main：拉取最新更改
git push origin main：推送更改
git config --global user.name "your_name"：设置全局用户名
git config --global user.email "your_email"：设置全局邮箱
十一、今日学习总结与问题解决方案
（一）学习内容
今日重点学习了：
Python 基础语法（变量、运算符、语句、函数、模块、类、装饰器、文件操作）。
Git 版本控制（仓库初始化、文件管理、远程仓库连接与同步）。
（二）遇到的问题及解决方案
1. 科学上网导致git push失败
问题描述：使用科学上网工具时，git push操作因端口冲突失败，报错：
fatal: unable to access 'https://github.com/yourusername/yourrepo.git/': Failed to connect to github.com port 443: Connection refused

解决方案：
临时关闭科学上网：确保网络直接连接，避免代理干扰。
指定代理端口（如需保留代理）：
git config --global http.proxy http://127.0.0.1:1080
git config --global https.proxy https://127.0.0.1:1080

（端口号需与代理工具设置一致）
2. 远程仓库无法同步空文件夹
问题描述：本地空文件夹无法推送到远程仓库，Git 忽略空目录。
解决方案：
在文件夹内创建任意文件（如.gitkeep或README.md），确保 Git 识别目录。
检查.gitignore文件，避免目标目录被排除。
3. Python 基础问题
变量类型转换错误：
问题：使用int("abc")导致ValueError。
解决：先校验输入格式，或用try-except捕获异常：
try:
    num = int(input("Enter a number: "))
except ValueError:
    print("请输入有效数字！")

作用域冲突：
问题：函数内修改全局变量未声明global。
解决：明确声明global或nonlocal关键字：
x = 10
def update_x():
    global x  # 声明全局变量
    x += 5

文件操作路径错误：
问题：文件路径不存在或权限不足。
解决：使用绝对路径或确保路径存在，推荐用with语句自动管理文件句柄：
with open("./data/example.txt", "r") as f:  # 相对路径更安全
    content = f.read()

十二、总结
今日通过理论学习与实践操作，掌握了 Python 基础语法和 Git 版本控制的核心操作。通过解决实际问题（如网络代理冲突、Git 同步机制、Python 作用域规则），加深了对知识的理解。后续需进一步练习复杂函数设计、Git 分支管理等进阶内容，巩固学习成果。

