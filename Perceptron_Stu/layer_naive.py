# 乘法层
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        
        out = x * y
        return out
    
    def backward(self, dout):
        dx = dout * self.y #翻转
        dy = dout * self.x
        
        return dx, dy
    
class AddLayer:
    def __init__(self):
        pass
    
    def forward(self, x0, x1):
        out = x0 + x1
        return out
    
    def backward(self, dout):
        dx0 = dout * 1
        dx1 = dout * 1
        
        return dx0, dx1