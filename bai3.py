def is perfect(n):
    sum = 0
    for i in range (1,n):
        if(n % i == 0):
           sum = sum +i
    return (sum==n)
n = int(input('nhap n '))
if (is_perfect(n)):
    print(n,'is perfect number')
else:
    print(n,'not perfectnumber')