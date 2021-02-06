import time

def timer(f):
    def wrapper():
        ini = time.time()
        f()
        elapsed = time.time() - ini
        print(f'Total {elapsed}s elapsed for execution.')
    return wrapper