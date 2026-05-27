a = 5
print(a)

def test(a):
    print(a)
    test(10)
    test(5)
    return a

test(12)
