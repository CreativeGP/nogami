import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.util import tracefunc

sys.setprofile(tracefunc)

def foo(a):
    print("foo ", a)
print("hello")
foo(3)