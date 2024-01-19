import numpy

num_list =  [12, "xin chao", -5]
for num in num_list:
    print(num)

for index in range(0,3,2):
    print(num_list[index])

def Greeting(name, question = " How are you?"):
    print("Hi ", name, question)

Greeting("Quang")

def counting():
    return 1, 2, 3

num1, num2, num3 = counting()
print(num1, num3)

list1 = [2, -1, 7, 12]
print(list1[0:4])


arr1 = numpy.array([2, -1, 7])
print(arr1[1:3])
