import time

def timer(f):
    def wrapper(*args, **kwargs):
        ini = time.time()
        res = f(*args, **kwargs)
        elapsed = time.time() - ini
        print(f'Total {elapsed}s elapsed for {f.__name__} execution.')
        return res
    return wrapper