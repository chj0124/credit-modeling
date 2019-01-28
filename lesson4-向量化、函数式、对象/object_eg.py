"""
对象
"""


#%%  定义对象
class Human:
    def __init__(self):
        self.gender = 'male'
        self.name = 'Trump'

    def sum(self):
        print("大家好，我是 " + self.name)


#%% 生成实例
Trump = Human()
Trump.sum()
Trump.gender
