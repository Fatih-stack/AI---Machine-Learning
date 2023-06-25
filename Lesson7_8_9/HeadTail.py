import random

HEAD = 1

def head_tail(n):
    head = tail = 0

    for _ in range(n):
        val = random.randint(0, 1)
        if val == HEAD:
            head += 1
        else:
            tail += 1
            
    return head / n, tail / n
            
        
head, tail = head_tail(10)
print(f'head = {head}, tail = {tail}')        

head, tail = head_tail(100)
print(f'head = {head}, tail = {tail}') 

head, tail = head_tail(1000)
print(f'head = {head}, tail = {tail}') 

head, tail = head_tail(10_000)
print(f'head = {head}, tail = {tail}') 

head, tail = head_tail(100_000)
print(f'head = {head}, tail = {tail}') 

head, tail = head_tail(1_000_000)
print(f'head = {head}, tail = {tail}') 

head, tail = head_tail(10_000_000)
print(f'head = {head}, tail = {tail}') 

head, tail = head_tail(100_000_000)
print(f'head = {head}, tail = {tail}') 