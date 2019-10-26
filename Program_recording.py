#  切换位置, 使用cd命令
C:\Users\lztia>cd C:\Users\lztia\Desktop\Python Learning

# 学习计算成绩提升比例并输出

grade_1 = 72
grade_2 = 85
inc = (85-72)/72*100
print('Your grade increased %.1f%%' % inc)


# 学习列表List

L = [
	['Apple', 'Google', 'Microsoft'],
	['Java', 'Python', 'Ruby', 'PHP'],
	['Adam', 'Bart', 'Lisa']
]
print(L[0][0]) #打印apple


# 学习元组tuple

t = ('a', 'b', ['A', 'B'])
t[2][0] = 'X'
t[2][1] = 'Y'
print(t)


# 学习输入输出及if判断

age1 = input('please input your age:')
age = int(age1)
if age >= 18:
	print('You are an Adult')
elif age >= 12:
	print('You are a Teenager')
else:
	print('You are a Child')


# 作业 判断BMI

Height = input('请输入你的身高(cm):')
Weight = input('请输入你的体重(kg):')
H = float(Height)
W = float(Weight)
BMI = W/(H*H)
print('\n') 
print('你的BMI指数为:', '%.2f' % BMI)
if BMI > 32:
	print('严重肥胖')
elif BMI > 28:
	print('肥胖')
elif BMI > 25:
	print('过重')
elif BMI > 18.5:
	print('正常')
elif BMI > 0:
	print('过轻')
else:
	print('输入数据错误')


# 学习range和list

sum = 0
for x in range(101):
	sum = sum + x
print('sum of 1-100:', sum)
a = list(range(5)) #list函数
print(a)


# while循环

L = ['Bart', 'Lisa', 'Adam']
n = 2
while n >= 0:
	print('Hello,', L[n])
	n = n - 1


# break结束当前循环 - 输出1-10之间所有数

n = 1
while n <= 100:
	if n > 10: #if为判断语句
		break #break结束当前循环
	print(n)
	n = n + 1
print('END')


# continue跳过当前循环，继续下一循环 - 输出10以内奇数

n = 0
while n < 10:
	n = n + 1
	if n % 2 == 0:
		continue
	print(n)
print('END')

# 如果代码进入死循环，按ctrl+c可以强制终止运行


# 字典学习 - dict

d = {'Mike': 65, 'Lucy': 32, 'Tracy':45}
d['Mike']

d.get('Lucy') #判断Lucy是否在字典里，不在会返回none

'Thomas' in d #判断Thomas是否在字典里

d['Jake'] = 80 #向dict内放入值

d.pop('Tracy') #删除key
d


# set学习，set没有对应value，不会出现重复的key
s = ([1,2,3]) #用list集合引入set作为key
s.add(4) #将4加入set中
s.remove(4) #将4从set中移除

# set取交集和并集
s1 = set([1,2,3])
s2 = set([2,3,4])
s1 & s2  #交集，输出结果为{2,3}
s1 | s2  #并集，输出结果为{1,2,3,4}

#不可变对象, replace操作只针对'abc', 不会改变'abc'的内容。会生成新的字符串
a = 'abc'
b = a.replace('a','A')
print('b:', b) #输出Abc
print('a:', a) #输出abc

# 字符串截取
https://www.cnblogs.com/xunbu7/p/8074417.html


# 将（1,2,3）分别放入dict和set中,均正确。因为（1,2,3）是tuple, 不可变

In [35]: a = (1,2,3)
    ...: b = (1,[2,3])
    ...: d = {a:10}
    ...: d
Out[35]: {(1, 2, 3): 10}

In [36]: a = (1,2,3)
    ...: b = (1,[2,3])
    ...: d = set(a)
    ...: d
Out[36]: {1, 2, 3}

# 将(1,[2,3])分别放入dict和set中，均报错。 因为[2,3]为list，可随时添加删除，可变

# 函数学习及调用
abs(-12.13) # 绝对值函数
max(1,2,3,4) #查找最大值
a = abs #将函数赋值给变量，简化函数调用

#类型转换
int('123') 输出123
int(12.34) #输出12
float('12.34') #输出12.34
str(1.23) #输出‘1.23’
bool(1) #输出True
bool('') #输出False


#练习，用hex()把函数转化成16进制的字符串
n1 = 255
n2 = 1000
print('%d的十六进制为%s \n %d的十六进制为%s' % (n1, hex(n1), n2, hex(n2)))


# 定义绝对值函数

def my_abs(x):
	if not isinstance(x, (int, float)):
		raise TypeError('bad operated type') # 判断输入值类型
	if x >= 0:
		return x
	else:
		return -x
print(my_abs(-99))

from abstest import my_abs  #导入函数my_abs，需要在目标文件夹内打开python


# 引入math函数

import math

def move(x, y, step, angle=0):
	nx = x + step * math.cos(angle)
	ny = y - step * math.sin(angle)
	return nx, ny

r = move(100, 100, 60, math.pi / 6)
print(r)
# 函数的返回结果为tuple，(151.96152422706632, 70.0)


#函数练习 计算一元二次方程的两个解

import math

def quadratic(a, b, c):
	temp = b * b - (4 * a * c)
	if temp < 0:
		return ('无解') # 方程无解时的输出
	else:
		temp1 = math.sqrt(temp)
		x1 = (- b + temp1) / (2 * a)
		x2 = (- b - temp1) / (2 * a)
		return x1, x2

print('quadratic(20, 1, 1) =', quadratic(20, 1, 1))

# 添加函数默认值
def power(x, n=2):
	s = 1
	while n > 0:
		n = n - 1
		s = s * x
	return s

# 定义可变参数
def calc(*number):
	sum = 0
	for n in number:
		sum = sum + n * n
	return sum
calc(1,2,3,4) # 输入可变参数进行函数运算

nums = [1,2,3,4]
calc(*nums) # 将list转为可变参数

# 定义关键字参数, 可输入多个扩展信息
def person(name, age, **kw):
	print('name:', name, 'age:', age, 'other:', kw)

>>>person('adam', 40, gender='M', job='engineer')
name: adam age: 40 other: {'gender': 'M', 'job': 'engineer'}

# 可变参数练习，将输入的数全部相乘
def product(x, *y):
	y1 = 1
	for n in y:
		y1 = y1 * n
	return x * y1


# 使用了尾递归
def fact(n):
	return fact_iter(n, 1)

def fact_iter(num, product):
	if num == 1:
		return product
	return fact_iter(num - 1, num * product)


#练习题，汉诺塔
def move(n, a, b, c):
	if n == 1:
		print(a, '->' , c)
	else:
		move(n-1, a, c, b)
		move(1, a, b, c)
		move(n-1, b, a, c)

#切片操作
>>> L = ['Michael', 'Sarah', 'Tracy', 'Bob', 'Jack']
>>> L[0:3]
['Michael', 'Sarah', 'Tracy'] # 从0开始索引，到3截止，不包括3
>>> L[1:3]
['Sarah', 'Tracy']  # 从1开始索引，到3截止，不包括3
>>> L[-2:-1]
['Bob']  # 倒数第一个元素为-1

#  切片练习，去除输入元素的首尾空格
# 自己写的程序
def trim(s):
	n = 0
	m = len(s)

	if m == 0:
		s = ''
		return s

	while s[n] == ' ':
		n = n + 1
		if n == m:
			s = ''
			return s

	while s[m - 1] == ' ':
		m = m - 1

	s = s[n:m]

	return s

# 最简单的写法，[:1]和[-1:]为其子集，可以判断为空集，不会报错
def trim(s):
	while s[:1] == ' ':
		s = s[1:]
	while s[-1:] == ' ':
		s = s[:-1]
	return s


# 测试
if trim('hello  ') != 'hello':
    print('hello  测试失败!')
elif trim('  hello') != 'hello':
    print('  hello测试失败!')
elif trim('  hello  ') != 'hello':
    print('  hello  测试失败!')
elif trim('  hello  world  ') != 'hello  world':
    print('  hello  world  测试失败!')
elif trim('') != '':
    print('测试失败!')
elif trim('    ') != '':
    print('     测试失败!')
else:
    print('测试成功!')

# 迭代(for顺序迭代)作业，判断list中最大最小值

def findMinAndMax(L):
	if len(L) == 0:
		return (None, None)
	Lmax = L[0]
	Lmin = L[0]
	for i in L:
		if i > Lmax:
			Lmax = i
		if i < Lmin:
			Lmin = i
	return (Lmin, Lmax)

if findMinAndMax([]) != (None, None):
    print('none none测试失败!')
elif findMinAndMax([7]) != (7, 7):
    print('7 测试失败!')
elif findMinAndMax([7, 1]) != (1, 7):
    print('7 1测试失败!')
elif findMinAndMax([7, 1, 3, 9, 5]) != (1, 9):
    print('71395测试失败!')
else:
    print('测试成功!')




#  list生成列表
In [27]: list(range(1,11))
Out[27]: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

#  for循环直接生成列表
In [25]: [x * x for x in range(1, 11)]
Out[25]: [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# for循环加上if判断
In [28]: [x * x for x in range(1,11) if x % 2 ==0]
Out[28]: [4, 16, 36, 64, 100]

# 双循环生成全排列
In [1]: [m + n for m in 'ABC' for n in 'XYZ']
Out[1]: ['AX', 'AY', 'AZ', 'BX', 'BY', 'BZ', 'CX', 'CY', 'CZ']

# dict里面的items()同时迭代key和value
In [2]: d = {'x': 'A', 'y': 'B', 'z': 'C'}
   ...: for k, v in d.items():
   ...:     print(k, '=', v)
   ...:
x = A
y = B
z = C

# 列表生成式迭代key和value
In [3]: d = {'x': 'A', 'y': 'B', 'z': 'C'}
   ...: [k + '=' + v for k, v in d.items()]
Out[3]: ['x=A', 'y=B', 'z=C']

# list中所有字符串变小写
In [4]: L = ['Hello', 'World', 'IBM', 'Apple']
   ...: [s.lower() for s in L]
Out[4]: ['hello', 'world', 'ibm', 'apple']

# 测试，添加if语句让表达式顺利通过
L1 = ['Hello', 'World', 18, 'IBM', None]
L3 = []
for x in L1:
	if isinstance(x, str): # 判断函数是否为字符串
		L3.append(x)
L2 = [s.lower() for s in L3]

# 另一种方法，更简单
L1 = ['Hello', 'World', 18, 'IBM', None]
L2 = [a.lower() for a in L1 if isinstance(a,str)]

In [10]: print(L2)
['hello', 'world', 'ibm']

# 斐波拉契数列函数
def fib(max):
	n, a, b = 0, 0, 1
	while n < max:
		print(b)
		a, b = b, a + b
		n = n + 1
	return 'done'

# 斐波拉契数列生成器，可以使用next()来一个个打印
def fib(max):
	n, a, b = 0, 0, 1
	while n < max:
		yield b
		a, b = b, a + b
		n = n + 1
	return 'done'

In [9]: for n in fib(6):  # 使用for循环来打印生成器
   ...:     print(n)
   ...:
1
1
2
3
5
8

#  作业，使用生成器来打印杨辉三角（方法1，投机取巧）
def triangles():
	L = []
	M = [1]
	a = 1
	while 1:
		yield M
		a = a + 1
		L = [L[i] + L[i+1] for i in range(a-2)]  #计算出中间需要累加的数
		L.insert(0, 1)  #首项插入1
		L.insert(a-1, 1)  #尾项插入1
		M = L

###
### 高阶函数，包含map/reduce, filter, sorted 学习内容 ###
###

#  高阶函数，函数引用函数
def add(x, y, f):
	return(f(x) + f(y))

In [8]: print(add(-5, 6, abs))
11

# map()函数的运用
In [1]: def f(x):
   ...:     return x * x
   ...:
In [2]: r = map(f,[1,2,3,4,5,6,7,8,9])
In [3]: list(r)
Out[3]: [1, 4, 9, 16, 25, 36, 49, 64, 81]

#  map()高阶函数的用法，一行搞定换算
In [6]: list(map(str, [1,2,3,4,5,6,7,8,9]))
Out[6]: ['1', '2', '3', '4', '5', '6', '7', '8', '9']

#  reduce()函数的使用: reduce(f, [x1, x2, x3, x4]) = f(f(f(x1, x2), x3), x4)
In [10]: def fn(x, y):
    ...:     return x * 10 + y
In [11]: reduce(fn, [1,2,3,4,5])
Out[11]: 12345

# 作业，用map()写标准名转化，首字母大写，其他小写
def normalize(name):
	name = name.lower()
	return name[0].upper() + name[1:]

L1 = ['adam', 'LISA', 'barT']
L2 = list(map(normalize, L1))
print(L2)

# 作业，用reduce()写一个函数，累乘所有数
def mtp(x, y):
	return x * y

def prod(L):
	return reduce(mtp, L)

print('3 * 5 * 7 * 9 =', prod([3, 5, 7, 9]))
if prod([3, 5, 7, 9]) == 945:
    print('测试成功!')
else:
    print('测试失败!')


#  作业，用map和reduce写一个函数，把字符串‘123.456’转换成浮点数123.456

def str2float(s):
	# 计算小数点在第几位
	n = 0
	numb = ''
	for i in s:
		if i == '.':
			len_int = n
		else:
			numb = numb + i
			n = n + 1
	len_float = n

	# 函数，用于合成整数
	def multp(x, y):
		return x * 10 + y

	# 函数，将字符串转化为整数
	def str2int(a):
		digits = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, 
		'7': 7, '8': 8, '9': 9}
		return digits[a]

	# 最终输出结果，先输出一个长整数，然后除以10的小数点后的位数次方
	return reduce(multp, map(str2int, numb))/(10**(len_float - len_int))

#  作业解法2，使用split函数分割小数点
	from functools import reduce

def str2float(s):

	def multp(x, y):
		return x * 10 + y

	def str2int(a):
		digits = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, 
		'7': 7, '8': 8, '9': 9}
		return digits[a]

	return reduce(multp, map(str2int, s.split('.')[0])) + reduce(multp, map(str2int, s.split('.')[1]))/10**(len(s.split('.')[1]))


#  filter函数，过滤掉所有结果为false的值
def is_odd(n):
	return n % 2 == 1
In [41]: list(filter(is_odd, [1, 2, 4, 5, 6, 9, 10, 15]))
Out[41]: [1, 5, 9, 15]

#  filter例题，生成素数
def _odd_iter():
	n = 1
	while True:
		n = n + 2
		yield n

def _not_divisible(n):
	return lambda x: x % n > 0

def prime():
	yield 2
	it = _odd_iter()
	while True:
		n = next(it)
		yield n
		it = filter(_not_divisible(n), it)
# 最后需要使用for循环打印特定范围内的素数


#  作业，筛选出回数，及左右对称的数
def fun(s):
	n = 0
	str_s = str(s)
	a = len(str_s) // 2
	len_s = len(str_s)

	if len_s == 1:
		return s
	else:
		while n < a:
			if str_s[n] == str_s[len_s - 1 - n]:
				n = n + 1
			else:
				return 0
		return s

def is_palindrome(n):
	return fun(n) > 0

# 大神一行代码写完
def is_palindrome(n):
	return str(n) == str(n)[::-1]   #双冒号，表示顺序相反操作


# 测试:
output = filter(is_palindrome, range(1, 1000))
print('1~1000:', list(output))
if list(filter(is_palindrome, range(1, 200))) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 22, 33, 44, 55, 66, 77, 88, 99, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191]:
    print('测试成功!')
else:
    print('测试失败!')


# 作业，按姓名排序
L = [('Bob', 75), ('Adam', 92), ('Bart', 66), ('Lisa', 88)]

def by_name(t):
	return t[0]

L2 = sorted(L, key=by_name)
print(L2)
>>>[('Adam', 92), ('Bart', 66), ('Bob', 75), ('Lisa', 88)]

# 作业，按成绩从高到低排序
def by_score(t):
	return -t[1]

L2 = sorted(L, key=by_score)
print(L2)
>>>[('Adam', 92), ('Lisa', 88), ('Bob', 75), ('Bart', 66)]


#  返回函数，每次调用都引用了变量i，但没有立刻执行，所以自后返回三个值都是9
def count():
    fs = []
    for i in range(1, 4):
        def f():
             return i*i
        fs.append(f)
    return fs

f1, f2, f3 = count()

>>> f1()
9
>>> f2()
9
>>> f3()
9
# 返回函数不要饮用任何循环变量，或者后续会发生变化的变量


def count():
    def f(j):
        def g():
            return j*j
        return g
    fs = []

    for i in range(1, 4):
        fs.append(f(i)) # f(i)立刻被执行，因此i的当前值被传入f()
    return fs

# 作业，计数器——用generator的写法
def creatCounter():
	def counter():
		i = 0
		while True:
			i = i + 1
			yield i
	f = counter()
	def g():
		return next(f)
	return g

# 作业，计数器——用nonlocal变量的写法
def creatCounter():
	i = 0
	def counter():
		nonlocal i
		i = i + 1
		return i
	return counter

counterA = creatCounter()
print(counterA(), counterA(), counterA(), counterA(), counterA())
>>> 1 2 3 4 5

#  作业，匿名函数lambda的用法

L = list(filter(lambda n: n % 2 == 1, range(1, 20)))
print(L)
>>> [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]


# 作业，装饰器decorator

import time, functools

def metric(fn):
	@functools.wraps(fn)

	def wrapper(*args, **kw):
		time_start = time.time()
		func = fn(*args, **kw)
		time_end = time.time()
		print('%s executed in %s ms' % (fn.__name__, (time_start - time_end)))
		return func
	
	return wrapper

# 测试
@metric
def fast(x, y):
    time.sleep(0.0012)
    return x + y;

@metric
def slow(x, y, z):
    time.sleep(0.1234)
    return x * y * z;

f = fast(11, 22)
s = slow(11, 22, 33)
if f != 33:
    print('测试失败!')
elif s != 7986:
    print('测试失败!')

# 作业1，写一个装饰器，在函数调用前后打印begin call和end call
def log(f):
	@functools.wraps(f)

	def wrapper(*args, **kw):
		print('%s() begin call' %f.__name__)
		func = f(*args, **kw)
		print('%s() end call' %f.__name__)
		return func

	return wrapper

@log
def f():
	print('abc')

f()

# int base的使用, 输入12为8进制，转化成10进制整数
int('12', 8)
>>> 10

#偏函数的使用
import time, functools
int2 = functools.partial(int, base=2)
int2('10000')
>>> 16

#  创建类（class)
class Student(object):
	def __init__(self, name, score):
		self.name = name
		self.score = score

	def get_grade(self):
		if self.score >= 90:
			return 'A'
		elif self.score >=60:
			return 'B'
		else:
			return 'C'

lisa = Student('Lisa', 99)
bart = Student('Bart', 59)
print(lisa.name, lisa.get_grade())
print(bart.name, bart.get_grade())


# 访问限制作业
#  把gender字段对外隐藏，并用get_gender()和set_gender()代替，并检查参数有效性
class Student(object):
	def __init__(self, name, gender):
		self.__name = name
		self.__gender = gender

	def get_gender(self):
		return self.__gender

	def set_gender(self, gender):
		if gender == 'male' or gender == 'female':
			self.__gender = gender
		else:
			raise ValueError('error gender')

#  子类自动集成父类的模块
class Animal(object):
	def run(self):
		print('Animal is running...')

class Dog(Animal):
	pass

class Cat(Animal):
	pass

dog = Dog()

dog.run()
>>>Animal is running...

#  子类的run()会覆盖父类的run(), 代码可以实现多态
class Animal(object):
	def run(self):
		print('Animal is running...')

class Dog(Animal):
	def run(self):
		print('Dog is running...')

	def eat(self):
		print('Eat meat...')

class Cat(Animal):
	def run(self):
		print('Cat is running...')

dog = Dog()
cat = Cat()

dog.run()
cat.run()

>>>Dog is running...
>>>Cat is running...

#  鸭子类型，只要有run()的定义，就可以直接运行相关功能。
class Animal(object):
	def run(self):
		print('Animal is running...')

class Dog(Animal):
	def run(self):
		print('Dog is running...')

class Cat(Animal):
	def run(self):
		print('Cat is running...')

class Timer(object):
	def run(self):
		print('Start...')

def run_twice(animal):
	animal.run()
	animal.run()

run_twice(Timer())
Start...
Start...

run_twice(Animal())
Animal is running...
Animal is running...

run_twice(Cat())
Cat is running...
Cat is running...

# type()判断对象类型
In [9]: type(123)
Out[9]: int

In [10]: type('str')
Out[10]: str

In [11]: type(None)
Out[11]: NoneType

In [12]: type(abs)
Out[12]: builtin_function_or_method

In [13]: type(cat)
Out[13]: __main__.Cat

# type()返回class类型
In [14]: type(123) == int
Out[14]: True

In [15]: type('abc') == str
Out[15]: True

In [16]: type('abc') == type('123')
Out[16]: True

# type判断是否为函数
import types
def fn():
	pass

type(fn) == types.FunctionType
>>>True

# isinstance()的使用
In [24]: a = Animal()
    ...: d = Dog()
    ...: h = Husky()

In [25]: isinstance(h, Husky)
Out[25]: True

In [26]: isinstance(h, Dog)
Out[26]: True

In [27]: isinstance(d, Husky)
Out[27]: False

In [28]: isinstance('a', str)
Out[28]: True

# 判断是否是其中某一类型的一种
In [29]: isinstance([1,2,3], (list, tuple))
Out[29]: True

#  getattr(), hasattr(), setattr()的用法
class MyObject(object):
	def __init__(self):
		self.x = 9
	def power(self):
		return self.x * self.x

obj = MyObject()

    ...: hasattr(obj, 'x') # 有属性'x'吗？
Out[31]: True

In [32]: hasattr(obj, 'y') # 有属性'y'吗？
Out[32]: False

In [33]: setattr(obj, 'y', 19) # 设置一个属性'y'

In [34]: hasattr(obj, 'y')
Out[34]: True

In [35]: getattr(obj, 'y') # 获取属性‘y’
Out[35]: 19

In [36]: obj.y # 获取属性‘y’
Out[36]: 19

In [37]: getattr(obj, 'z', 404) # 获取属性'z'，如果不存在，返回默认值404
Out[37]: 404

In [38]: hasattr(obj, 'power') # 有属性’power‘吗？
Out[38]: True


#  作业，每增加一个学生实例，人数统计+1
class Student(object):
	count = 0

	def __init__(self, name):
		self.name = name
		Student.count = Student.count + 1

# 测试:
if Student.count != 0:
    print('测试失败!')
else:
    bart = Student('Bart')
    if Student.count != 1:
        print('测试失败!')
    else:
        lisa = Student('Bart')
        if Student.count != 2:
            print('测试失败!')
        else:
            print('Students:', Student.count)
            print('测试通过!')

# __slots__限制class可以添加实例的属性

class Student(object):
	__slots__ = ('name', 'age')

s = Student()
s.name = 'Michael'
s.age = 25
s.score = 99  # 这里会报错，因为score不在__slots__中.

class GraduateStudent(Student):
	pass

g = GraduateStudent()
g.score = 99  # 子类的属性定义不受父类__slots__的影响 
print(g.score)
>>> 99


# 作业，使用@property装饰器，直接可以进行属性调用

class Screen(object):

	@property
	def width(self):
		return self._width

	@width.setter
	def width(self, wid_value):
		self._width = wid_value

	@property
	def height(self):
		return self._height
	
	@height.setter
	def height(self, hei_value):
		self._height = hei_value

	@property
	def resolution(self):
		return self._width * self._height
	
	# 测试:
s = Screen()
s.width = 1024
s.height = 768
print('resolution =', s.resolution)
if s.resolution == 786432:
    print('测试通过!')
else:
    print('测试失败!')


# 多重继承
class Mammal(object):
	pass

class Runnable(object):
	def run(self):
		print('running...')

class Dog(Mammal, Runnable):
	pass

#  MixIn设计，即多重继承设计

#  定制类 __str__, 可以直接打印出想要的字符串形式
#  __repr__, 不使用print情况下，直接输入时引用的地址

class Student(object):
	def __init__(self, name):
		self.name = name
	def __str__(self):
		return "Student object (name: %s)" % self.name
	__repr__ = __str__  # 不使用print，直接输入变量也可得到同样输出

print(Student('Mike'))
>>> Student object (name: Mike)

# __iter__的使用

class Fib(object):
	def __init__(self):
		self.a, self.b = 0, 1
	
	def __iter__(self):
		return self # 实例本身就是迭代对象，返回自己

	def __next__(self):
		self.a, self.b = self.b, self.a + self.b
		if self.a > 100:
			raise StopIteration() #退出循环的条件
		return self.a  #返回下一个值

for n in Fib():
	print(n)
>>>
1
2
3
...
89

# __getitem__的使用

class Fib(object):
	def __getitem__(self, n):
		a, b = 1, 1
		for n in range(n):
			a, b = b, a + b
		return a

f = Fib()
f[10]
>>> 89

# __getattr__的使用

class Student(object):
	
	def __init__(self):
		self.name = 'Mike'

	def __getattr__(self, attr):
		if attr == 'score':
			return 99
		raise AttributeError('\'Student\' object has no attribute \'s\'' % attr)

#  __call__的使用, 可以直接对实例进行调用
class Student(object):
	
	def __init__(self, name):
		self.name = name

	def __call__(self):
		print('My name is %s' % self.name)

s = Student('Steve')
s()
>>> My name is Steve


# 枚举法的应用
from enum import Enum

Month = Enum('Month', ('Jan', 'Feb', 'Mar'))

for name, member in Month.__members__.items():
	print(name, '=>', member, ',', member.value)

Jan => Month.Jan , 1
Feb => Month.Feb , 2
Mar => Month.Mar , 3


# unique装饰器检查是否有重复值

from enum import Enum, unique

@unique
class Weekday(Enum):
	Sun = 0
	Mon = 1
	Tue = 2
	Wed = 2

>>> ValueError: duplicate values found in <enum 'Weekday'>: Wed -> Tue


#####   使用元类: type(), metaclass, 以后需要的时候再学习 #####


# try...except...finally...机制

try:
	print('try...')
	r = 10 / 0
	print('result', r)
except ZeroDivisionError as e: #try出现错误，直接跳过后面程序，执行except
	print('except:', e)
finally:
	print('finally...')
print('END')

try...
except: division by zero
finally...
END 

# 修改为正确执行语句之后的输出
try:
	print('try...')
	r = 10 / 2
	print('result', r)
except ZeroDivisionError as e:
	print('except:', e)
finally:
	print('finally...')
print('END')

try...
result 5.0
finally...
END

# 使用多个except来捕获error
try:
	print('try...')
	r = 10 / int('2')
	print('result', r)
except ValueError as e:
	print('ValueError:', e)
except ZeroDivisionError as e:
	print('ZeroDivisionError:', e)
else:
	print('no error!')
finally:
	print('finally...')
print('END')


# 使用logging记录错误信息
import logging

def foo(s):
    return 10 / int(s)

def bar(s):
    return foo(s) * 2

def main():
	try:
    	bar('0')
	except Exception as e:
    	logging.exception(e)

main()
print('END')

>>>
ROR:root:division by zero
Traceback (most recent call last):
  File "<ipython-input-2-cc27997fe30e>", line 11, in main
    bar('0')
  File "<ipython-input-2-cc27997fe30e>", line 7, in bar
    return foo(s) * 2
  File "<ipython-input-2-cc27997fe30e>", line 4, in foo
    return 10 / int(s)
ZeroDivisionError: division by zero
END


# 单元测试

import unittest

class Student(object):
    def __init__(self, name, score):
        self.name = name
        self.score = score
    def get_grade(self):
        if self.score >= 60 and self.score < 80:
            return 'B'
        if self.score >= 80 and self.score <= 100:
            return 'A'
        if self.score >= 0 and self.score < 60:
            return 'C'
        raise ValueError

class TestStudent(unittest.TestCase):

    def test_80_to_100(self):
        s1 = Student('Bart', 80)
        s2 = Student('Lisa', 100)
        self.assertEqual(s1.get_grade(), 'A')
        self.assertEqual(s2.get_grade(), 'A')

    def test_60_to_80(self):
        s1 = Student('Bart', 60)
        s2 = Student('Lisa', 79)
        self.assertEqual(s1.get_grade(), 'B')
        self.assertEqual(s2.get_grade(), 'B')

    def test_0_to_60(self):
        s1 = Student('Bart', 0)
        s2 = Student('Lisa', 59)
        self.assertEqual(s1.get_grade(), 'C')
        self.assertEqual(s2.get_grade(), 'C')

    def test_invalid(self):
        s1 = Student('Bart', -1)
        s2 = Student('Lisa', 101)
        with self.assertRaises(ValueError):
            s1.get_grade()
        with self.assertRaises(ValueError):
            s2.get_grade()

if __name__ == '__main__':
    unittest.main()

# 运行之后的结果如下
....
----------------------------------------------------------------------
Ran 4 tests in 0.000s

OK

***Repl Closed***


# doctest方法，运行出现错误的话，就会执行注释中的程序

def fact(n):
    '''
    Calculate 1*2*...*n
    
    >>> fact(1)
    1
    >>> fact(10)
    3628800
    >>> fact(-1)
    Traceback (most recent call last):
        ...
    ValueError
    '''
    if n < 1:
        raise ValueError()
    if n == 1:
        return 1
    return n * fact(n - 1)

if __name__ == '__main__':
    import doctest
    doctest.testmod()

open, read()来打开读取文件
f = open('C:/Users/lztia/Desktop/Python Learning/ppt.txt', 'r')
f.read()
>>> 'Hello World'
f.close()

# 使用with来打开读取文件，并自动关闭
with open('/path/to/file', 'r') as f:
    print(f.read())

# 读取图片文件二进制
In [6]: f = open('C:/Users/lztia/Desktop/Python Learning/pict.jpg', 'rb')
   ...: f.read()

# 读取gbk编码文件，同事忽略存在非法编码的字符
>>> f = open('/Users/michael/gbk.txt', 'r', encoding='gbk', errors='ignore')

# 使用write来写入
f = open('C:/Users/lztia/Desktop/Python Learning/ppt.txt', 'w')
f.write('Test Writing')
f.close()

# 使用with来写入文件，并自动关闭
with open('C:/Users/lztia/Desktop/Python Learning/ppt.txt', 'w') as f:
    f.write('Hello, world!')

# StringIO往内存中读写str，用getvalue()获得写入后的str
from io import StringIO
f = StringIO()
f.write('hello')
f.write(' ')
f.write('world!')
print(f.getvalue())

>>>hello world!

# 读取StringIO，用str初始化StringIO，然后读取
from io import StringIO
f = StringIO('Hello!\nHi!\nGoodbye!')
while True:
    s = f.readline()
    if s == '':
        break
    print(s.strip())
>>>
Hello!
Hi!
Goodbye!

# 内存中读写二进制数据
from io import BytesIO
f = BytesIO()
f.write('中文'.encode('utf-8'))
print(f.getvalue())
>>> b'\xe4\xb8\xad\xe6\x96\x87'

# pickle序列化，把任意文件序列化成bytes，然后就可以写入文件
In [2]: import pickle
   ...: d = dict(name='Bob', age=20, score=88)
   ...: pickle.dumps(d)
Out[2]: b'\x80\x03}q\x00(X\x04\x00\x00\x00nameq\x01X\x03\x00\x00\x00Bobq\x02X\x03\x00\x00\x00ageq\x03K\x14X\x05\x00\x00\x00scoreq\x04KXu.'

# 用pickle.dump()把对象直接写入一个file-like object
f = open('dump.txt', 'wb')
pickle.dump(d, f)
f.close()

#使用pickle.load()反序列化出对象
f = open('dump.txt', 'rb')
d = pickle.load(f)
f.close()
d
>>> {'name': 'Bob', 'age': 20, 'score': 88}

# JSON将python的对象转化为JSON对象
import json
d = dict(name='Bob', age=20, score=88)
json.dumps(d)
 >>>'{"name": "Bob", "age": 20, "score": 88}'

#JSON反序列化撑Python对象
json_str = '{"name": "Bob", "age": 20, "score": 88}'
json.loads(json_str)
>>> {'name': 'Bob', 'age': 20, 'score': 88}


# 用class表示对象，然后序列化成JSON

import json

class Student(object):
	def __init__(self, name, age, score):
		self.name = name
		self.age = age
		self.score = score

def student2dict(std):  # 用于将Student转化成可序列化的dict
	return {
	'name': std.name,
	'age': std.age,
	'score': std.score
	}

s = Student('Bob', 20, 88)
print(json.dumps(s, default=student2dict))

>>> {"name": "Bob", "age": 20, "score": 88}

# 输出中文的ASCII码
import json
obj = dict(name='小明', age=20)
s = json.dumps(obj, ensure_ascii=True)
print(s)
>>> {"name": "\u5c0f\u660e", "age": 20}

# 输出中文
import json
obj = dict(name='小明', age=20)
s = json.dumps(obj, ensure_ascii=False)
print(s)
>>> {"name": "小明", "age": 20}

# 多进程

from multiprocessing import Process
import os

def run_proc(name):
	print('Run child process %s (%s)...' % (name, os.getpid()))

if __name__ == '__main__':
	print('Parent process %s.' % os.getpid())
	p = Process(target=run_proc, args=('test',))
	print('Child process will start.')
	p.start()
	p.join()
	print('Child process end')

>>>
Parent process 2300.
Child process will start.
Run child process test (7832)...
Child process end

#  Pool多进程，join()方法会等所有子进程执行完毕，调用前先要调用close()

from multiprocessing import Pool
import os, time, random

def long_time_task(name):
	print('Run task %s (%s)...' % (name, os.getpid()))
	start = time.time()
	time.sleep(random.random() * 3)
	end = time.time()
	print('Task %s runs %0.2f seconds.' %(name, (end-start)))

if __name__ == '__main__':
	print('Parent process %s.' % os.getpid())
	p = Pool(4) #如果改成5，则不会有等待效果
	for i in range(5):
		p.apply_async(long_time_task, args=(i,))
	print('Waiting for all subprocesses done...')
	p.close()
	p.join()
	print('All subprocesses done.')

>>>
Parent process 19540.
Waiting for all subprocesses done...
Run task 0 (17720)...
Run task 1 (13724)...
Run task 2 (14896)...
Run task 3 (20892)...
Task 2 runs 0.22 seconds.
Run task 4 (14896)...
Task 3 runs 0.52 seconds.
Task 1 runs 0.94 seconds.
Task 4 runs 1.12 seconds.
Task 0 runs 2.40 seconds.
All subprocesses done.


# 子进程，在python中运行nslookup www.python.org

import subprocess

print('$ nslookup www.python.org')
r = subprocess.call(['nslookup', 'www.python.org'])
print('Exit code:', r)
>>>
$ nslookup www.python.org
Server:  HG6Box
Address:  192.168.1.1

Non-authoritative answer:
Name:    dualstack.python.map.fastly.net
Addresses:  2a04:4e42:2e::223
          151.101.196.223
Aliases:  www.python.org

Exit code: 0

# 多线程，使用lock，将程序锁定成单线程工作，避免参数被调用混乱
import time, threading

balance = 0

def change_it(n):
    global balance
    balance = balance + n
    balance = balance - n
lock = threading.Lock()

def run_thread(n):
	for i in range(100000):
		lock.acquire()
		try:
			change_it(n)
		finally:
			lock.release()

t1 = threading.Thread(target=run_thread, args=(5,))
t2 = threading.Thread(target=run_thread, args=(8,))
t1.start()
t2.start()
t1.join()
t2.join()
print(balance)


# 多线程，使用ThreadLocal生成可供全局调用的局部变量。
# local_school是一个全局变量，但每个属性local_school.student是线程的局部变量

import threading

local_school = threading.local()

def process_student():
	std = local_school.student
	print('Hello, %s (in %s)' % (std, threading.current_thread().name))

def process_thread(name):
	local_school.student = name
	process_student()

t1 = threading.Thread(target=process_thread, args=('Alice',), name='Thread-A')
t2 = threading.Thread(target=process_thread, args=('Bob',), name='Thread-B')
t1.start()
t2.start()
t1.join()
t2.join()


# 分布式进程，详情参考教程
# https://blog.kasora.moe/2016/06/12/python-%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%A1%E7%AE%97/

# task_master.py

import time,queue
from multiprocessing.managers import BaseManager
from multiprocessing import freeze_support
#任务个数
task_number = 10;
#定义收发队列
task_queue = queue.Queue(task_number);
result_queue = queue.Queue(task_number);
def gettask():
    return task_queue;
def getresult():
     return result_queue;
def test():
    #windows下绑定调用接口不能使用lambda，所以只能先定义函数再绑定
    BaseManager.register('get_task',callable = gettask);
    BaseManager.register('get_result',callable = getresult);
    #绑定端口并设置验证码，windows下需要填写ip地址，linux下不填默认为本地
    manager = BaseManager(address = ('127.0.0.1',5002),authkey = b'123');
    #启动
    manager.start();
    try:
        #通过网络获取任务队列和结果队列
        task = manager.get_task();
        result = manager.get_result();
        #添加任务
        for i in range(task_number):
            print('Put task %d...' % i)
            task.put(i);
        #每秒检测一次是否所有任务都被执行完
        while not result.full():
            time.sleep(1);
        for i in range(result.qsize()):
            ans = result.get();
            print('task %d is finish , runtime:%d s' % ans);
    
    except:
        print('Manager error');
    finally:
        #一定要关闭，否则会爆管道未关闭的错误
        manager.shutdown();
        
if __name__ == '__main__':
    #windows下多进程可能会炸，添加这句可以缓解
    freeze_support()
    test();

# task_worker

import time, sys, queue, random
from multiprocessing.managers import BaseManager

BaseManager.register('get_task')
BaseManager.register('get_result')

conn = BaseManager(address = ('127.0.0.1',5002), authkey = b'123');

try:
    conn.connect();

except:
    print('连接失败');
    sys.exit();
    
task = conn.get_task();
result = conn.get_result();

while not task.empty():
    n = task.get(timeout = 1);
    print('run task %d' % n);
    sleeptime = random.randint(0,3);
    time.sleep(sleeptime);
    rt = (n, sleeptime);
    result.put(rt);

if __name__ == '__main__':
    pass;

# 分布式进程结束


# 正则表达式
import re

test = '010-12345'
if re.match(r'^\d{3}\-\d{3,8}$', test):
	print('ok')
else:
	print('failed')

# 切分字符串，无论多少空格都可以正常分割
import re
re.split(r'\s+', 'a b   c')
>>> ['a', 'b', 'c']

# 切分字符串，忽略空格和逗号
In [5]: import re
   ...: re.split(r'[\s\,]+', 'a, b, c   d')
Out[5]: ['a', 'b', 'c', 'd']

# 正则表达式，分组
>>> m = re.match(r'^(\d{3})-(\d{3,8})$', '010-12345')
>>> m
<_sre.SRE_Match object; span=(0, 9), match='010-12345'>
>>> m.group(0)
'010-12345'
>>> m.group(1) # 第一组
'010'
>>> m.group(2) # 第二组
'12345'

# 正则对时间识别分组
>>> t = '19:05:30'
>>> m = re.match(r'^(0[0-9]|1[0-9]|2[0-3]|[0-9])\:(0[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|[0-9])\:(0[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|[0-9])$', t)
>>> m.groups()
('19', '05', '30')


# 正则表达式作业，判断email地址合法性
import re

def is_valid_email(addr):
	if re.match(r'[0-9a-zA-Z\.]+@[0-9a-zA-Z]+\.com', addr):
		return True
	else:
		return False
# 测试:
assert is_valid_email('someone@gmail.com')
assert is_valid_email('bill.gates@microsoft.com')
assert not is_valid_email('bob#example.com')
assert not is_valid_email('mr-bob@example.com')
print('ok')
>>> ok

# 作业2，提取邮件中姓名
import re

def name_of_email(addr):
	return re.match(r'.*?(\w\.]+[\s\w]+|[\w\s]+)', addr).group(1)

# 测试:
assert name_of_email('<Tom Paris> tom@voyager.org') == 'Tom Paris'
assert name_of_email('tom@voyager.org') == 'tom'
print('ok')

# 获取当前日期时间
from datetime import datetime
now = datetime.now()
print(now)
>>> 2019-09-23 19:27:51.841394

#获取指定日期和时间
from datetime import datetime
dt = datetime(2015, 4, 19, 12, 20)
print(dt)
>>> 2015-04-19 12:20:00

# 把时间转换成时间戳timestamp
In [11]: from datetime import datetime
    ...: dt = datetime(2015, 4, 19, 12, 20)
    ...: dt.timestamp()
Out[11]: 1429471200.0

# 把时间戳转化成datetime
from datetime import datetime
t = 1429417200.0
print(datetime.fromtimestamp(t))
>>> 2015-04-18 21:20:00

# 转换成UTC时间
from datetime import datetime
t = 1429417200.0
print(datetime.fromtimestamp(t))
print(datetime.utcfromtimestamp(t))
>>>
2015-04-18 21:20:00
2015-04-19 04:20:00

# str转换为datetime，注意年份为大写Y
# https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior

from datetime import datetime
cday = datetime.strptime('2015-6-1 18:19:59', '%Y-%m-%d %H:%M:%S')
print(cday)
>>> 2015-06-01 18:19:59

# datetime转换成str
from datetime import datetime
now = datetime.now()
print(now.strftime('%a, %b, %d %H:%M'))
>>> Mon, Sep, 23 20:04

# datetime加减
from datetime import datetime, timedelta
now = datetime.now()
now
now + timedelta(hours=10)
now - timedelta(days=1)
now + timedelta(days=2, hours=12)

# 强制设置时区为UTC +0:00
from datetime import datetime, timedelta, timezone
utc_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
print(utc_dt)
>>> 2019-09-24 03:59:11.587029+00:00
# 转换时区为北京时间
bj_dt = utc_dt.astimezone(timezone(timedelta(hours=8)))
print(bj_dt)
>>> 2019-09-24 12:04:30.982365+08:00
# 转换时区为东京时间
tokyo_dt = utc_dt.astimezone(timezone(timedelta(hours=9)))
print(tokyo_dt)
>>> 2019-09-24 13:09:27.114172+09:00
# 将北京时间转换为东京时间
tokyo_dt2 = bj_dt.astimezone(timezone(timedelta(hours=9)))
print(tokyo_dt2)
>>> 2019-09-24 13:09:27.114172+09:00


#  作业，将时间信息和时区信息转化成timestamp

import re
from datetime import datetime, timedelta, timezone

def to_timestamp(dt_str, tz_str):
	date_time = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
	time_1 = re.match(r'UTC((\+|\-)\d+):\d\d', tz_str).group(1)
	time_1_int = int(time_1)
	time_zone = timezone(timedelta(hours=time_1_int))
	time_2 = date_time.replace(tzinfo=time_zone)
	time_stamps = time_2.timestamp()
	return time_stamps

# 测试:
t1 = to_timestamp('2015-6-1 08:10:30', 'UTC+7:00')
assert t1 == 1433121030.0, t1

t2 = to_timestamp('2015-5-31 16:10:30', 'UTC-09:00')
assert t2 == 1433121030.0, t2
>>>
print('ok')

# namedtuple来表示坐标
from collections import namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(1,2)
p.x
>>> 1
p.y
>>> 2

# deque可以快速的实现插入和删除操作的双向列表

from collections import deque

q = deque(['a', 'b', 'c'])
q.append('x') # 往右插入x
q.appendleft('y') # 往左插入y
q
>>> deque(['y', 'a', 'b', 'c', 'x'])
q.pop() # 从右边删去一位
>>> 'x' 
q
>>> deque(['y', 'a', 'b', 'c'])

q.popleft() # 从左边删去一位
>>> 'y'
q
>>> deque(['a', 'b', 'c'])

# defaultdict返回默认值
from collections import defaultdict
dd = defaultdict(lambda:'N/A')
dd['key1'] = 'abc'
dd['key1']
>>> 'abc'
dd['key2'] # key2不存在，返回默认值
>>> 'N/A'

# 按照key的插入顺序进行排序， OrderedDict
# 可以实现FIFO，先进先出的dict
from collections import OrderedDict

od = OrderedDict()
od['z'] = 1
od['y'] = 2
od['x'] = 3
list(od.keys())
>>> ['z', 'y', 'x']

# Counter计数器，可以拿来统计字符出现次数
from collections import Counter
c = Counter()
for ch in 'programming':
	c[ch] = c[ch] +1
c
>>> Counter({'p': 1, 'r': 2, 'o': 1, 'g': 2, 'a': 1, 'm': 2, 'i': 1, 'n': 1})

# base64
>>> import base64
>>> base64.b64encode(b'binary\x00string')
b'YmluYXJ5AHN0cmluZw=='
>>> base64.b64decode(b'YmluYXJ5AHN0cmluZw==')
b'binary\x00string'

# urlsafe转化方式，把+和/替换成-和_
>>> base64.b64encode(b'i\xb7\x1d\xfb\xef\xff')
b'abcd++//'
>>> base64.urlsafe_b64encode(b'i\xb7\x1d\xfb\xef\xff')
b'abcd--__'

# 作业，写一个能处理去掉=的base64解码函数，因为base64在编码过程中，会将==删除
import base64

def safe_base64_decode(s):
	digi_length = len(s)
	if digi_length%4 == 0:
		return base64.b64decode(s)
	else:
		rem = 4 - digi_length%4
		s1 = s + b'='*rem
		return base64.b64decode(s1)
# 测试:
assert b'abcd' == safe_base64_decode(b'YWJjZA=='), safe_base64_decode('YWJjZA==')
assert b'abcd' == safe_base64_decode(b'YWJjZA'), safe_base64_decode('YWJjZA')
print('ok')

# struct的pack函数把任意数据类型变成bytes：
import struct
struct.pack('>I', 10240099) # >表示网络序，I表示4字节无符号整数
>>> b'\x00\x9c@c'

# unpack把bytes变成相应数据类型
# I表示4字节无符号整数，H表示2字节无符号整数
import struct
struct.unpack('>IH', b'\xf0\xf0\xf0\xf0\x80\x80')
Out[66]: (4042322160, 32896)

# unpack来读取bmp的前30个字节
In [67]: s = b'\x42\x4d\x38\x8c\x0a\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00\x80\x02\x00\x00\x68\x01\x00\x00
    ...: \x01\x00\x18\x00'

In [68]: struct.unpack('<ccIIIIIIHH',s)
Out[68]: (b'B', b'M', 691256, 0, 54, 40, 640, 360, 1, 24)
# b'B',b'M'说明是windows位图，位图大小是640*360，颜色是24

# 识别是否是bmp图像，并且打印出图像的长宽高和颜色
import base64, struct
bmp_data = base64.b64decode('Qk1oAgAAAAAAADYAAAAoAAAAHAAAAAoAAAABABAAAAAAADICAAASCwAAEgsAAAAAAAAAAAAA/3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9/AHwAfAB8AHwAfAB8AHwAfP9//3//fwB8AHwAfAB8/3//f/9/AHwAfAB8AHz/f/9//3//f/9//38AfAB8AHwAfAB8AHwAfAB8AHz/f/9//38AfAB8/3//f/9//3//fwB8AHz/f/9//3//f/9//3//f/9/AHwAfP9//3//f/9/AHwAfP9//3//fwB8AHz/f/9//3//f/9/AHwAfP9//3//f/9//3//f/9//38AfAB8AHwAfAB8AHwAfP9//3//f/9/AHwAfP9//3//f/9//38AfAB8/3//f/9//3//f/9//3//fwB8AHwAfAB8AHwAfAB8/3//f/9//38AfAB8/3//f/9//3//fwB8AHz/f/9//3//f/9//3//f/9/AHwAfP9//3//f/9/AHwAfP9//3//fwB8AHz/f/9/AHz/f/9/AHwAfP9//38AfP9//3//f/9/AHwAfAB8AHwAfAB8AHwAfAB8/3//f/9/AHwAfP9//38AfAB8AHwAfAB8AHwAfAB8/3//f/9//38AfAB8AHwAfAB8AHwAfAB8/3//f/9/AHwAfAB8AHz/fwB8AHwAfAB8AHwAfAB8AHz/f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//3//f/9//38AAA==')

bmp_data1 = bmp_data[:30]
def bmp_info(data):
	bmp_info = struct.unpack('<ccIIIIIIHH', bmp_data1)
	if bmp_info[0] == b'B' and bmp_info[1] == b'M': # 识别是否为BMP图
		bmp_w = bmp_info[6]
		bmp_h = bmp_info[7]
		bmp_c = bmp_info[9]
		return {'width': bmp_w, 'height': bmp_h, 'color': bmp_c}
	else:
		return(print('Not BMP'))

# 测试
bi = bmp_info(bmp_data)
assert bi['width'] == 28
assert bi['height'] == 10
assert bi['color'] == 16
print('ok')

# 使用MD5摘要算法，计算一个字符串的MD5值
import hashlib

md5 = hashlib.md5()
md5.update('how to use md5 in python hashlib?'.encode('utf-8'))
print(md5.hexdigest())
>>> d26a53750bc40b38b65a520292f69306

# 使用SHA1摘要算法
import hashlib

sha1 = hashlib.sha1()
sha1.update('how to use sha1 in '.encode('utf-8'))
sha1.update('python hashlib?'.encode('utf-8'))
print(sha1.hexdigest())

# 作业，使用MD5验证用户名和密码正确性
import hashlib

db = {
    'michael': 'e10adc3949ba59abbe56e057f20f883e',
    'bob': '878ef96e86145580c38c87f0410ad153',
    'alice': '99b1c2188db85afee403b1536010c2c9'
}

def login(user, password):
	password_md5 = hashlib.md5()
	password_md5.update(password.encode('utf-8'))
	if user in db:
		if db[user] == password_md5.hexdigest():
			return True
		return False

# 测试:
assert login('michael', '123456')
assert login('bob', 'abc999')
assert login('alice', 'alice2008')
assert not login('michael', '1234567')
assert not login('bob', '123456')
assert not login('alice', 'Alice2008')
print('ok')

# 作业，验证加盐后的用户名和密码
import hashlib, random

def get_md5(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()

class User(object):
    def __init__(self, username, password):
        self.username = username
        self.salt = ''.join([chr(random.randint(48, 122)) for i in range(20)])
        self.password = get_md5(password + self.salt)
db = {
    'michael': User('michael', '123456'),
    'bob': User('bob', 'abc999'),
    'alice': User('alice', 'alice2008')
}

def login(username, password):
	if username in db:
		user = db[username]
		if user.password == get_md5(password + user.salt):
			return True
		return False
# 更简洁的写法
#def login(username, password):
# 	user = db[username]
# 	return user.password == get_md5(password + user.salt)

# 测试:
assert login('michael', '123456')
assert login('bob', 'abc999')
assert login('alice', 'alice2008')
assert not login('michael', '1234567')
assert not login('bob', '123456')
assert not login('alice', 'Alice2008')
print('ok')

# hmac算法，需要提供口令的哈希算法
import hmac
message = b'Hello, world!'
key = b'secret'
h = hmac.new(key, message, digestmod='MD5')
h.hexdigest()
>>> 'fa4ee7d173f2d97ee79022d1a7355bcf'

# 作业，使用hmac算法验证用户名和密码
import hmac, random

def hmac_md5(key, s):
    return hmac.new(key.encode('utf-8'), s.encode('utf-8'), 'MD5').hexdigest()

class User(object):
    def __init__(self, username, password):
        self.username = username
        self.key = ''.join([chr(random.randint(48, 122)) for i in range(20)])
        self.password = hmac_md5(self.key, password)

db = {
    'michael': User('michael', '123456'),
    'bob': User('bob', 'abc999'),
    'alice': User('alice', 'alice2008')
}

def login(username, password):
	user = db[username]
	return user.password == hmac_md5(user.key, password)

# 测试:
assert login('michael', '123456')
assert login('bob', 'abc999')
assert login('alice', 'alice2008')
assert not login('michael', '1234567')
assert not login('bob', '123456')
assert not login('alice', 'Alice2008')
print('ok')


# 迭代器，无限序列，通过takewhile截取其中一段
import itertools
natuals = itertools.count(1)
ns = itertools.takewhile(lambda x: x<=10, natuals)
list(ns)
>>> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 不然就会无限迭代下去，只能使用ctrl+c强制终止
>>> natuals = itertools.count(1)
>>> for n in natuals:
...     print(n)
...
1
2
3
...

# chain()把迭代对象串联起来，形成更大的迭代器
import itertools
for c in itertools.chain('ABC', 'XYZ'):
	print(c)
>>>
A
B
C
X
Y
Z

# groupby()把迭代器中相邻的重复元素挑出来放一起
for key, group in itertools.groupby('AAABBBCCAAA'):
	print(key, list(group))
>>>
A ['A', 'a', 'a']
B ['B', 'B', 'b']
C ['c', 'C']
A ['A', 'A', 'a']

# 忽略大小写排序
for key, group in itertools.groupby('AaaBBbcCAAa', lambda c: c.upper()):
	print(key, list(group))
>>>
A ['A', 'a', 'a']
B ['B', 'B', 'b']
C ['c', 'C']
A ['A', 'A', 'a']

# 作业，计算圆周率
import itertools

def pi(N):

    # step 1: 创建一个奇数序列: 1, 3, 5, 7, 9, ...
	natuals = itertools.count(1, 2)
    # step 2: 取该序列的前N项: 1, 3, 5, 7, 9, ..., 2*N-1.
	ns = itertools.takewhile(lambda x: x<=2*N-1, natuals)
    # step 3: 添加正负符号并用4除: 4/1, -4/3, 4/5, -4/7, 4/9, ...
    he = 0
    x = 1
    # step 4: 求和:
    for n in ns:
    	he = he + 4*x/n
    	x = -x
    return he

print(pi(10))
print(pi(100))
print(pi(1000))
print(pi(10000))
assert 3.04 < pi(10) < 3.05
assert 3.13 < pi(100) < 3.14
assert 3.140 < pi(1000) < 3.141
assert 3.1414 < pi(10000) < 3.1415
print('ok')

#  urllib的request模块，抓取URL内容

from urllib import request

with request.urlopen('https://www.easy-mock.com/mock/5cbec5d8bfb3b05625e96633/dreamlf/urllibTest') as f:
	data = f.read()
	print('Status:', f.status, f.reason)
	for k, v in f.getheaders():
		print('%s: %s' % (k, v))
	print('Data:', data.decode('utf-8'))
>>>
Status: 200 OK
Server: Tengine
Date: Sun, 29 Sep 2019 20:16:33 GMT
Content-Type: application/json; charset=utf-8
Content-Length: 165
Connection: close
X-Request-Id: 913f6953-0ce2-4bbc-9f2c-e45993955750
Vary: Accept, Origin
Rate-Limit-Remaining: 1
Rate-Limit-Reset: 1569788194
Rate-Limit-Total: 2
Data: {"message":"此接口只为廖雪峰大神python教程里面的练习题使用，没有其他意义","query":{"results":{"channel":{"location":{"city":"Beijing"}}}}}

# 模拟iphone6浏览器发送GET请求
from urllib import request

req = request.Request('http://douban.com/')
req.add_header('User-Agent', 'Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25')
with request.urlopen(req) as f:
	print('Status:', f.status, f.reason)
	for k, v in f.getheaders():
		print('%s: %s' % (k, v))
	print('Data:', f.read().decode('utf-8'))

>>>
...
    <meta name="viewport" content="width=device-width, user-scalable=no, initial-scale=1.0, minimum-scale=1.0, maximum-scale=1.0">
    <meta name="format-detection" content="telephone=no">
    <link rel="apple-touch-icon" sizes="57x57" href="http://img4.douban.com/pics/cardkit/launcher/57.png" />
...


# python使用HTMLParser解析HTML
# 爬虫类知识，需要后续学习加强

from html.parser import HTMLParser
from html.entities import name2codepoint

class MyHTMLParser(HTMLParser):

    def handle_starttag(self, tag, attrs):
        print('<%s>' % tag)

    def handle_endtag(self, tag):
        print('</%s>' % tag)

    def handle_startendtag(self, tag, attrs):
        print('<%s/>' % tag)

    def handle_data(self, data):
        print(data)

    def handle_comment(self, data):
        print('<!--', data, '-->')

    def handle_entityref(self, name):
        print('&%s;' % name)

    def handle_charref(self, name):
        print('&#%s;' % name)

parser = MyHTMLParser()
parser.feed('''<html>
<head></head>
<body>
<!-- test html parser -->
    <p>Some <a href=\"#\">html</a> HTML&nbsp;tutorial...<br>END</p>
</body></html>''')


# Pillow模块的使用，将图像缩放50%

from PIL import Image

# 打开jpg图像文件
im = Image.open('pict.jpg')
# 获得图像尺寸
w, h = im.size
print('Original image size: %sx%s' % (w, h))
# 缩放到50%
im.thumbnail((w//2, h//2))
print('Resize image to: %sx%s' % (w//2, h//2))
# 把缩放后的图像用jpeg格式保存
im.save('thumbnail.jpg', 'jpeg')


# Pillow生成四位验证码

from PIL import Image, ImageDraw, ImageFont, ImageFilter

import random

# 随机字母
def mdChar():
	return chr(random.randint(65,90))

# 随机颜色1
def mdColor():
	return(random.randint(64,255), random.randint(64, 255), random.randint(64, 255))

# 随机颜色2
def mdColor2():
	return(random.randint(32,127), random.randint(32,127), random.randint(32,127))

# 240 * 60:
width = 60 * 4
height = 60
image = Image.new('RGB', (width, height), (255, 255, 255))
# 创建Font对象
font = ImageFont.truetype('arial.ttf', 36)
# 创建Draw对象
draw = ImageDraw.Draw(image)
# 填充每个像素：
for x in range(width):
	for y in range(height):
		draw.point((x,y), fill=mdColor())
# 输出文字
for t in range(4):
	draw.text((60 * t + 10, 10), mdChar(), font=font, fill=mdColor2())
# 模糊
image = image.filter(ImageFilter.BLUR)
image.save('code.jpg', 'jpeg')

# 更多Pillow的功能参考官方文档
# https://pillow.readthedocs.io/en/stable/



# 使用requests，通过GET访问网站
import requests
r = requests.get('https://www.douban.com/')
r.status_code
>>> 200
r.text
>>> '<!DOCTYPE HTML>\n<html>\n<head>\n<meta name="description" content="提供图书、电影、音乐唱片的推荐、评论和...'

# 传入一个dict作为params来访问带参数的URL
r = requests.get('https://www.douban.com/search', params={'q': 'python', 'cat': '1001'})
r.url
>>> 'https://www.douban.com/search?q=python&cat=1001'


# chardet检测编码, confidence表明检测可靠率

import chardet

chardet.detect(b'Hello, world!')
>>> {'encoding': 'ascii', 'confidence': 1.0, 'language': ''}

data = '床前明月光，疑似地上霜'.encode('gbk')
chardet.detect(data)
>>> {'encoding': 'GB2312', 'confidence': 0.99, 'language': 'Chinese'}

# psutil模块，获取CPU信息
import psutil
psutil.cpu_count() # CPU逻辑数量
>>> 4
psutil.cpu_count(logical=False) # CPU物理核心
>>> 2

# 统计CPU的用户/系统/空闲时间
In [4]: psutil.cpu_times()
Out[4]: scputimes(user=114374.703125, system=146100.390625, idle=2000505.21875, interrupt=9259.875, dpc=3225.359375)

# 获取物理内存
In [5]: psutil.virtual_memory()
Out[5]: svmem(total=8488738816, available=2471723008, percent=70.9, used=6017015808, free=2471723008)

# 获取交换内存信息
In [6]: psutil.swap_memory()
Out[6]: sswap(total=17078673408, used=11670372352, free=5408301056, percent=68.3, sin=0, sout=0)

# 获取磁盘分区信息
In [7]: psutil.disk_partitions()
Out[7]: [sdiskpart(device='C:\\', mountpoint='C:\\', fstype='NTFS', opts='rw,fixed')]

# 获取磁盘使用情况
In [8]: psutil.disk_usage('/')
Out[8]: sdiskusage(total=241659351040, used=234372390912, free=7286960128, percent=97.0)

# 磁盘IO信息 
In [9]: psutil.disk_io_counters()
Out[9]: sdiskio(read_count=7161500, write_count=3975087, read_bytes=663593812992, write_bytes=198973404160, read_time=16849, write_time=34485)

# 获取网络读写字节/包的个数
psutil.net_io_counters()

psutil.net_if_addrs() # 获取网络接口信息

psutil.net_if_stats() # 获取网络接口状态

#获取当前网络连接信息
psutil.net_connections()

# 获取进程信息
>>> psutil.pids() # 所有进程ID
[3865, 3864, 3863, 3856, 3855, 3853, 3776, ..., 45, 44, 1, 0]
>>> p = psutil.Process(3776) # 获取指定进程ID=3776，其实就是当前Python交互环境
>>> p.name() # 进程名称
'python3.6'
>>> p.exe() # 进程exe路径
'/Users/michael/anaconda3/bin/python3.6'
>>> p.cwd() # 进程工作目录
'/Users/michael'
>>> p.cmdline() # 进程启动的命令行
['python3']
>>> p.ppid() # 父进程ID
3765
>>> p.parent() # 父进程
<psutil.Process(pid=3765, name='bash') at 4503144040>
>>> p.children() # 子进程列表
[]
>>> p.status() # 进程状态
'running'
>>> p.username() # 进程用户名
'michael'
>>> p.create_time() # 进程创建时间
1511052731.120333
>>> p.terminal() # 进程终端
'/dev/ttys002'
>>> p.cpu_times() # 进程使用的CPU时间
pcputimes(user=0.081150144, system=0.053269812, children_user=0.0, children_system=0.0)
>>> p.memory_info() # 进程使用的内存
pmem(rss=8310784, vms=2481725440, pfaults=3207, pageins=18)
>>> p.open_files() # 进程打开的文件
[]
>>> p.connections() # 进程相关网络连接
[]
>>> p.num_threads() # 进程的线程数量
1
>>> p.threads() # 所有线程信息
[pthread(id=1, user_time=0.090318, system_time=0.062736)]
>>> p.environ() # 进程环境变量
{'SHELL': '/bin/bash', 'PATH': '/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:...', 'PWD': '/Users/michael', 'LANG': 'zh_CN.UTF-8', ...}
>>> p.terminate() # 结束进程
Terminated: 15 <-- 自己把自己结束了

# GUI图形界面，编写一个可以电机quit退出的提示框

from tkinter import *

class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

    def createWidgets(self):
        self.helloLabel = Label(self, text='Hello, world!')
        self.helloLabel.pack()
        self.quitButton = Button(self, text='Quit', command=self.quit)
        self.quitButton.pack()

app = Application()
# 设置窗口标题:
app.master.title('Hello World')
# 主消息循环:
app.mainloop()

# 可以输入文本，点击按钮，弹出对话框
from tkinter import *

class Application(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

    def createWidgets(self):
        self.nameInput = Entry(self)
        self.nameInput.pack()
        self.alertButton = Button(self, text='Hello', command=self.hello)
        self.alertButton.pack()

    def hello(self):
    	name = self.nameInput.get() or 'world'
    	messagebox.showinfo('Message', 'Hello, %s' % name)

app = Application()
# 设置窗口标题:
app.master.title('Hello World')
# 主消息循环:
app.mainloop()

# 海龟画图，画一个长方形
from turtle import *

#设置笔刷宽度
width(4)

#前进
forward(200)

#右转90度
right(90)

#笔刷眼色
pencolor('red')

forward(100)
right(90)

pencolor('green')
forward(200)
right(90)

pencolor('blue')
forward(100)
right(90)

# 调用done()使窗口等待被关闭，否则立刻关闭窗口
done()

# 用turtle画5个五角星
from turtle import *

def drawStar(x, y):
	pu() # pen up
	goto(x, y)
	pd() # pen down
	seth(0) #setheading(angle)
	for i in range(5):
		fd(40) #forward
		rt(144) #right(angle)

for x in range(0, 250, 50):
	drawStar(x, 0)

done()

# 画一个二叉树

from turtle import *

# 设置色彩模式是RGB：
colormode(255)

lt(90)

lv = 6
l = 120
s = 45

width(lv)

# 初始化RGB颜色
r = 0
g = 0
b = 0
pencolor(r, g, b)

penup()
bk(l)
pendown()
fd(l)

def draw_tree(l, level):
	global r, g, b
	w = width()
	width(w * 3.0/4.0)
	r = r + 1
	g = g + 1
	b = b + 3
	pencolor(r % 200, g % 200, b % 200)

	l = 3.0/4.0 * l

	lt(s)
	fd(l)

	if level < lv:
		draw_tree(l, level + 1)
	bk(l)
	rt(2 * s)
	fd(l)

	if level < lv:
		draw_tree(l, level + 1)
	bk(l)
	lt(s)

	width(w)

speed("fastest")

draw_tree(l, 4)

done()


# TCP编程

import socket
import ssl
# 创建一个socket，但sina只允许HTTPS协议，所以该例子无法使用
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s = ssl.wrap_socket(socket.socket())
s.connect(('www.sina.com.cn', 443))
# 端口号是固定的, 80为Web服务标准端口，SMTP服务是25，FTP是21，小于1024的是
# Internet标准服务端口，大于1024的可以任意使用
s.send(b'GET / HTTP/1.1\r\nHost: www.sina.com.cn\r\nConnection: close\r\n\r\n')

buffer = []
while True:
	d = s.recv(1024)
	if d:
		buffer.append(d)
	else:
		break
data = b''.join(buffer)

s.close()

header, html = data.split(b'\r\n\r\n', 1)
print(header.decode('utf-8'))
with open('sina.html', 'wb') as f:
	f.write(html)


# 例子，自己电脑建立服务器，并创建客户端与服务器连接

# 服务器代码
import socket
import threading
import time

# 创建一个基于IPv4和TCP协议的socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s.bind(('127.0.0.1', 9999)) # 监听端口
s.listen(5) # 指定等待连接的最大数量
print('Waiting for connection...')

def tcplink(sock, addr):
	print('Accept new connection from %s:%s...' % addr)
	sock.send(b'Welcome!')
	while True:
		data = sock.recv(1024)
		time.sleep(1)
		if not data or data.decode('utf-8') == 'exit':
			break
		sock.send(('Hello, %s' % data.decode('utf-8')).encode('utf-8'))
	sock.close()
	print('Connection from %s:%s closed' % addr)

while True:
	sock, addr = s.accept() #接受一个新连接
	#创建新线程来处理TCP连接
	t = threading.Thread(target=tcplink, args=(sock, addr)) 
	t.start()
# 服务器程序会一直运行，需要ctrl+C强制退出程序

# 客户端代码
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 建立连接
s.connect(('127.0.0.1', 9999))
# 接收欢迎消息
print(s.recv(1024).decode('utf-8'))
for data in [b'Michael', b'Tracy', b'Sarah']:
	# 发送数据
	s.send(data)
	print(s.recv(1024).decode('utf-8'))
s.send(b'exit')
s.close


# 使用SMTP协议发送邮件

import smtplib

server = "smtp.sina.com"
fromaddr= "lztianjianfeng@sina.com" #须修改
toaddr = "37142431@qq.com" #须修改
msg = """
to:%s
from:%s
Hello,I am smtp server
""" %(toaddr,fromaddr)
s = smtplib.SMTP(server)
s.set_debuglevel(1)
s.login("lztianjianfeng@sina.com","****************") #16位验证密码
s.sendmail(fromaddr,toaddr,msg)

# 有主题，显示发件人收件人信息的邮件发送
from email import encoders
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr

import smtplib

def _format_addr(s):
	name, addr = parseaddr(s)
	return formataddr((Header(name, 'utf-8').encode(), addr))

from_addr = input('From: ')
password = input('Password: ')
to_addr = input('To: ')
smtp_server = input('SMTP server: ')

msg = MIMEText('hello, send by Python...', 'plain', 'utf-8')
msg['From'] = _format_addr('Python学习者<%s>' % from_addr)
msg['To'] = _format_addr('被测试者<%s>' % to_addr)
msg['Subject'] = Header('来自SMTP的问候...', 'utf-8').encode()

server = smtplib.SMTP(smtp_server, 25) #SMTP协议默认端口为25
server.set_debuglevel(1)
server.login(from_addr, password)
server.sendmail(from_addr, [to_addr], msg.as_string())
server.quit()


# 发送带有附件的邮件

from email import encoders
from email.header import Header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart #新增
from email.mime.base import MIMEBase #新增
from email.utils import parseaddr, formataddr

import smtplib

def _format_addr(s):
	name, addr = parseaddr(s)
	return formataddr((Header(name, 'utf-8').encode(), addr))

# from_addr = input('From: ')
# password = input('Password: ')
# to_addr = input('To: ')
# smtp_server = input('SMTP server: ')
from_addr = 'lztianjianfeng@sina.com'
password = '3991a22453a3ae4c'
to_addr = '37142431@qq.com'
smtp_server = 'smtp.sina.com'

# 邮件对象
msg = MIMEMultipart()
msg['From'] = _format_addr('Python学习者<%s>' % from_addr)
msg['To'] = _format_addr('被测试者<%s>' % to_addr)
msg['Subject'] = Header('来自SMTP的问候...', 'utf-8').encode()

# 邮件正文是MIMEText
msg.attach(MIMEText('send with file...', 'plain', 'utf-8'))

# 添加附件就是加上一个MIMEBase，从本地读取一个图片：
with open('C:/Users/lztia/Desktop/Python Learning/Pillow Learning/pict.jpg', 'rb') as f:
	# 设置福建的MIME和文件名，这里是jpg类型：
	mime = MIMEBase('image', 'jpg', filename='test.jpg')
	# 加上必要的头信息
	mime.add_header('Content-Disposition', 'attachement', filename='test.jpg')
	mime.add_header('Content-ID', '<0>')
	mime.add_header('X-Attachement-Id', '0')
	# 把附件的内容读进来：
	mime.set_payload(f.read())
	# 用Base64编码：
	encoders.encode_base64(mime)
	# 添加到MIMEMultipart:
	msg.attach(mime)

server = smtplib.SMTP(smtp_server, 25) #SMTP协议默认端口为25
server.set_debuglevel(1)
server.login(from_addr, password)
server.sendmail(from_addr, [to_addr], msg.as_string())
server.quit()


# 发送邮件正文中插入图片的邮件

from email import encoders
from email.header import Header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.utils import parseaddr, formataddr

import smtplib

def _format_addr(s):
	name, addr = parseaddr(s)
	return formataddr((Header(name, 'utf-8').encode(), addr))

# from_addr = input('From: ')
# password = input('Password: ')
# to_addr = input('To: ')
# smtp_server = input('SMTP server: ')
from_addr = 'lztianjianfeng@sina.com'
password = '3991a22453a3ae4c'
to_addr = '37142431@qq.com'
smtp_server = 'smtp.sina.com'

# 邮件对象
msg = MIMEMultipart()
msg['From'] = _format_addr('Python学习者<%s>' % from_addr)
msg['To'] = _format_addr('被测试者<%s>' % to_addr)
msg['Subject'] = Header('来自SMTP的问候...', 'utf-8').encode()

# 邮件正文是MIMEText，该部分以html格式插入图片，图片有cid:0开始编号
msg.attach(MIMEText('<html><body><h1>Hello</h1>' +
    '<p><img src="cid:0"></p>' +
    '</body></html>', 'html', 'utf-8'))

# 添加附件就是加上一个MIMEBase，从本地读取一个图片：
with open('C:/Users/lztia/Desktop/Python Learning/Pillow Learning/pict.jpg', 'rb') as f:
	# 设置福建的MIME和文件名，这里是jpg类型：
	mime = MIMEBase('image', 'jpg', filename='test.jpg')
	# 加上必要的头信息
	mime.add_header('Content-Disposition', 'attachement', filename='test.jpg')
	mime.add_header('Content-ID', '<0>')
	mime.add_header('X-Attachement-Id', '0')
	# 把附件的内容读进来：
	mime.set_payload(f.read())
	# 用Base64编码：
	encoders.encode_base64(mime)
	# 添加到MIMEMultipart:
	msg.attach(mime)

server = smtplib.SMTP(smtp_server, 25) #SMTP协议默认端口为25
server.set_debuglevel(1)
server.login(from_addr, password)
server.sendmail(from_addr, [to_addr], msg.as_string())
server.quit()


# 简易的图形窗口发送邮件

from email import encoders
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr
from tkinter import *
import tkinter.messagebox as messagebox
import smtplib

#图形窗口
class Useritfc(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()

    def createWidgets(self):
        #发送者邮箱
        self.emailLable = Label(self, text='Email:')
        self.emailLable.pack()
        self.emailInput = Entry(self)
        self.emailInput.pack()
        #邮箱密码
        self.passwordLable = Label(self, text='Password:')
        self.passwordLable.pack()
        self.passwordInput = Entry(self, show='*')
        self.passwordInput.pack()
        #接受者邮箱
        self.recieverLable = Label(self, text='reciever:')
        self.recieverLable.pack()
        self.recieverInput = Entry(self)
        self.recieverInput.pack()
        #发送smtp
        self.smtpLable = Label(self, text='SMTP:')
        self.smtpLable.pack()
        self.smtpInput = Entry(self)
        self.smtpInput.pack()
        #发送内容
        self.sendtextLable = Label(self, text='text:')
        self.sendtextLable.pack()
        self.sendtextInput = Entry(self)
        self.sendtextInput.pack()
        #确认按钮
        self.submitButton = Button(self, text='Submit', command=self.submit)
        self.submitButton.pack()

    def submit(self):
        s_email = self.emailInput.get()
        s_password = self.passwordInput.get()
        s_reciever = self.recieverInput.get()
        s_smtp = self.smtpInput.get()
        s_sendtext = self.sendtextInput.get()
        if s_email and s_password and s_reciever and s_smtp and s_sendtext:
            startsend(s_smtp, s_email, s_password, s_reciever, s_sendtext)
            messagebox.showinfo('Message', 'OK!')
            self.sendtextInput.delete(0, END)
        else:
            #填表出错弹窗
            messagebox.showinfo('Message', 'Please input all item correctly!')



def _format_addr(s):
    name, addr = parseaddr(s)
    return formataddr((Header(name, 'utf-8').encode(), addr))


def startsend(smtp, email, pswd, reciever, sendtext):
    msg = MIMEText(sendtext, 'html', 'utf-8')
    msg['From'] = _format_addr('JimmyPy <%s>' % email)
    msg['To'] = _format_addr('Admin <%s>' % reciever)
    msg['Subject'] = Header('Sup Bro!', 'utf-8').encode()

    server = smtplib.SMTP(smtp, 587) # SMTP协议默认端口是25
    server.starttls()
    server.set_debuglevel(1)
    server.login(email, pswd)
    server.sendmail(email, [reciever], msg.as_string())
    server.quit()


#启动窗口程序
app = Useritfc()
app.master.title('SMTP email sender')
app.mainloop()


# 例子，通过POP3收取邮件

from email.parser import Parser
from email.header import decode_header
from email.utils import parseaddr

import poplib

# 输入邮件地址，口令和POP3服务器地址：
email = 'lztianjianfeng@sina.com'
password = '3991a22453a3ae4c'
pop3_server = 'pop.sina.com'
# email = input('Email: ')
# password = input('Password: ')
# pop3_server = input('POP3 server: ')

# 文本邮件的内容也是str,还需要检测编码，
# 否则，非UTF-8编码的邮件都无法正常显示：
def guess_charset(msg):
    print('msg::%s' % msg)
    # 得到字符集
    charset = msg.get_charset()
    print('charset::%s' % charset)
    if charset is None:
        # lower:所有大写字符为小写
        content_type = msg.get('Content-Type', '').lower()
        print('content_type::%s' % content_type)
        # find:检测字符串中是否包含子字符串
        # 返回charset=头字符的位置
        pos = content_type.find('charset=')
        print('pos::', pos)
        if pos >= 0:
            # strip:移除字符串头尾指定的字符(默认为空格)
            charset = content_type[pos + 8:].strip()
    print('charset::%s' % charset)
    return charset

# 邮件的Subject或者Email中包含的名字都是经过编码后的str,要正常显示就必须decode
def decode_str(s):
	# 在不转换字符集的情况下解码消息头值,返回一个list
	value, charset = decode_header(s)[0]
	if charset:
		value = value.decode(charset)
	return value

# indent用于缩进显示：
def print_info(msg, indent=0):
    # 初始分析
    if indent == 0:
        # 遍历获取 发件人，收件人，主题
        for header in ['From', 'To', 'Subject']:
            # 获得对应的内容
            value = msg.get(header, '')
            # 有内容
            if value:
                # 如果是主题
                if header == 'Subject':
                    # 解码主题
                    value = decode_str(value)
                else:
                    # parseaddr：解析字符串中的email地址
                    hdr, addr = parseaddr(value)
                    # 解码主题
                    name = decode_str(hdr)
                    # 合成内容
                    value = u'%s <%s>' % (name, addr)
            print('%s%s：%s' % (' ' * indent, header, value))
    # 如果消息由多个部分组成，则返回True。
    if msg.is_multipart():
        # 返回list,包含所有的子对象
        parts = msg.get_payload()
        # enumerate将其组成一个索引序列，利用它可以同时获得索引和值
        for n, part in enumerate(parts):
            # 打印消息模块编号
            print('%s part %s' % (' ' * indent, n))
            # 打印分割符号
            print('%s--------------------' % (' ' * indent))
            # 递归打印
            print_info(part, indent + 1)
    else:
        # 递归结束条件，打印最基本模块
        # 返回消息的内容类型。
        content_type = msg.get_content_type()
        # 如果是text类型或者是html类型
        if content_type == 'text/plain' or content_type == 'text/html':
            # 返回list,包含所有的子对象，开启解码
            content = msg.get_payload(decode=True)
            # 猜测字符集
            charset = guess_charset(msg)
            # 字符集不为空
            if charset:
                # 解密
                content = content.decode(charset)
            # 打印内容
            print('%s Text: %s' % (' ' * indent, content + '...'))
        else:
            print('%s Attachment: %s' % (' ' * indent, content_type))

# 连接POP3服务器
server = poplib.POP3(pop3_server)
# 可以打开或关闭调试信息：
server.set_debuglevel(1)
# 可选：打印POP3服务器的欢迎文字
print(server.getwelcome().decode('utf-8'))

# 身份认证
server.user(email)
server.pass_(password)

# stat()返回邮件数量和占用空间
print('Message: %s. Size: %s' % server.stat())
# list()返回所有邮件的编号：
resp, mails, octets = server.list()
# 可以查看返回的列表类似[b'1 82923', b'2 2184'...]
print(mails)

# 获取最新一封邮件，注意索引号从1开始
index = len(mails)
resp, lines, octets = server.retr(index)

# lines存储了邮件原始文本的每一行
# 可以获得整个邮件的原始文本
msg_content = b'\r\n'.join(lines).decode('utf-8')
# 稍后解析出邮件
msg = Parser().parsestr(msg_content)

# 可以根据邮件索引号直接从服务器删除邮件：
# server.dele(index)
print_info(msg)
# 关闭链接：
server.quit()

# 结果如下
+OK sina pop3 server ready
*cmd* 'USER lztianjianfeng@sina.com'
*cmd* 'PASS 3991a22453a3ae4c'
*cmd* 'STAT'
*stat* [b'+OK', b'6', b'23598']
Message: 6. Size: 23598
*cmd* 'LIST'
[b'1 3091', b'2 3849', b'3 3199', b'4 3625', b'5 3706', b'6 6128']
*cmd* 'RETR 6'
From：lztianjianfeng@hotmail.com <lztianjianfeng@hotmail.com>
To：lztianjianfeng@sina.com <lztianjianfeng@sina.com>
Subject：Hello
 part 0
--------------------
msg::Content-Type: text/plain; charset="iso-8859-1"
Content-Transfer-Encoding: quoted-printable

Greeting from hotmail address

charset::None
content_type::text/plain; charset="iso-8859-1"
pos:: 12
charset::"iso-8859-1"
  Text: Greeting from hotmail address
...
 part 1
--------------------
msg::Content-Type: text/html; charset="iso-8859-1"
Content-Transfer-Encoding: quoted-printable

<html>
<head>
<meta http-equiv=3D"Content-Type" content=3D"text/html; charset=3Diso-8859-=
1">
<style type=3D"text/css" style=3D"display:none;"> P {margin-top:0;margin-bo=
ttom:0;} </style>
</head>
<body dir=3D"ltr">
<div style=3D"font-family: Calibri, Helvetica, sans-serif; font-size: 12pt;=
 color: rgb(0, 0, 0);">
Greeting from hotmail address</div>
</body>
</html>

charset::None
content_type::text/html; charset="iso-8859-1"
pos:: 11
charset::"iso-8859-1"
  Text: <html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=iso-8859-1">
<style type="text/css" style="display:none;"> P {margin-top:0;margin-bottom:0;} </style>
</head>
<body dir="ltr">
<div style="font-family: Calibri, Helvetica, sans-serif; font-size: 12pt; color: rgb(0, 0, 0);">
Greeting from hotmail address</div>
</body>
</html>
...
*cmd* 'QUIT'