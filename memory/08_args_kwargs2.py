def test_args_kwargs(arg1, arg2, arg3):
    print("arg1:", arg1)
    print("arg2:", arg2)
    print("arg3:", arg3)


# *args는 입력 데이터를 tuple로 만들어 하나를 넣는다.
args = ("two", 3, 5)
test_args_kwargs(*args)
print("")

# **kwarg는 입력 데이터를 dictionary로 만들어 하나를 넣는다.
kwargs = {"arg3": 3, "arg2": "two", "arg1": 5}
test_args_kwargs(**kwargs)


'''
C:\Users\user\Anaconda3\envs\torch4\python.exe D:/pytorch===/torch_4/처음/args_kwargs2.py
arg1: two
arg2: 3
arg3: 5

arg1: 5
arg2: two
arg3: 3

Process finished with exit code 0
'''