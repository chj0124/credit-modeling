"""
对象
"""


#%%
class Human:
    def __init__(self):
        self.gender = 'male'
        self.name = 'Trump'

    def sum(self):
        print("大家好，我是 " + self.name)


#%%
Trump = Human()
Trump.sum()
