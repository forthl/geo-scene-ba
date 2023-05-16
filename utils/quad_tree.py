class Point:
    def __init__(self, x: int = 0, y : int = 0):
        self.x = x
        self.y = y


class Node:
    def __init__(self, pos: Point = None, data: float = 0.0):
        self.pos = pos
        self.data = data


class Quad:
    def __init__(self, top_l: Point = Point(), bot_R: Point = Point()):
        self.top_left = top_l
        self.bot_right = bot_R

        self.width = self.top_left.x + self.bot_right.x
        self.height = self.top_left.y + self.bot_right.y

        self.n = Node()

        self.top_left_tree = None
        self.top_right_tree = None
        self.bot_left_tree = None
        self.bot_right_tree = None

    def insert(self, node: Node):
        if not self.isBoundary(n.pos):
            return
        
        if abs(self.top_left.x - self.bot_right.x) <= 1 and abs(self.top_left.y - self.bot_right.y <= 1):
            if n is None:
                n = node
            return
        
        if self.width / 2 >= node.pos.x:
            # top left tree
            if self.height / 2 >= node.pos.y:
                if self.top_left_tree is None:
                    self.top_left_tree = Quad(
                        Point(self.top_left.x, self.top_left.y), 
                        Point(self.width / 2, self.height / 2)
                    )
                
                self.top_left_tree.insert(node)
            # bot left tree
            else:
                if self.bot_left_tree is None:
                    self.bot_left_tree = Quad(
                        Point(self.top_left.x, self.height / 2),
                        Point(self.width / 2, self.bot_right.y)
                    )
                    
                self.bot_left_tree.insert(node)
                
        else:
            #top right tree
            if self.height / 2 >= node.pos.y:
                if self.top_right_tree is None:
                    self.top_right_tree = Quad(
                        Point(self.width / 2, self.top_left.y),
                        Point(self.bot_right.x, self.height / 2)
                    )
                    
                self.top_right_tree.insert(node)
                
            else:
                if self.bot_right_tree is None:
                    self.bot_right_tree = Quad(
                        Point(self.width / 2,
                          self.height / 2),
                        Point(self.bot_right.x, self.bot_right.y)
                    )
                    
                self.bot_right_tree.insert(node)
                
    def search(self, p: Point):
        if not self.isBoundary(p):
            return None
        
        if self.n != None:
            return self.n
        
        if self.width / 2 >= p.x:
            # top left tree
            if self.height / 2 >= p.y:
                if self.top_left_tree is None:
                    return None
                return self.top_left_tree.search(p)
            # bot left tree
            else:
                if self.bot_left_tree is None:
                    return None
                return self.bot_left_tree.search(p)
            
        else:
            # top right tree
            if self.height / 2 >= p.y:
                if self.top_right_tree is None:
                    return None
                return self.top_right_tree.search(p)
            # bot right tree
            else:
                if self.bot_right_tree is None:
                    return None
                return self.bot_right_tree.search(p)
            
    def findInRadius(self, p: Point, radius):
        min_x = max(p.x - radius)
        max_x = max(p.x + radius)
        
        min_y = max(p.y - radius)
        max_y = max(p.y + radius)
        
        nodes = []
        
        for x in range(min_x, max_x):
            for y in range(min_y, max_y):
                n = self.search(Point(x, y))
                
                if n is not None:
                    nodes.append(n)
                    
        return nodes

    def isBoundary(self, p: Point):
        return p.x >= self.top_left.x and p.x <= self.bot_right.x and p.y >= self.top_left.y and p.y <= self.bot_right.y
