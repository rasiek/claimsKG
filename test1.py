

class Test:

    def __init__(self, param1, param2, param3) -> None:
        
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        # self.method1()
        self.__method2()

    def __method2(self):

        print('print private')


    def method1(self):

        print("print")


    def method2(self, *args, **kwargs):
        return super().method2(*args, **kwargs)
    



object1 = Test(1,2,3)

object1.method1()








