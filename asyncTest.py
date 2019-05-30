from concurrent.futures import ThreadPoolExecutor
import random
def hello(num):
    result = num[0] + num[1]
    for i in range(num[0]):
        result+=i
    return result
def go():
    pass
if __name__ == "__main__":
    # with ThreadPoolExecutor(max_workers=4) as executor:
        # for value in executor.map(hello, [(20500, 20),(4,3),(3,2),(2,1),(1,30)]):
            # print(value)
    # print('hi')
    li = [0,1,2,3]
    random.shuffle(li)
    print(li)
    random.shuffle(li)
    print(li)
    random.shuffle(li)
    print(li)