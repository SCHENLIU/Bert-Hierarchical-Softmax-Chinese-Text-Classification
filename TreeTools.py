# (value, subtrees)
class TreeTools:
    def __init__(self):
        # memoization for _count_nodes functions
        self._count_nodes_dict = {}
        self.label_dict = {
            "期货": 1,
            "沪深股票": 2,
            "银行": 3,
            "保险": 4,
            "信托": 5,
            "互联网金融": 6,
            "基金": 7,
            "外汇": 8,
            "贵金属": 9,
            "港股": 10,
            "债券": 11,
            "美股": 12,
            "行业新闻": 13,
            "二手车": 14,
            "改装/赛事": 15,
            "试驾评测": 16,
            "汽车导购": 17,
            "汽车金融": 18,
            "新车": 19,
            "行情报价": 20,
            "新能源汽车": 21,
            "汽车文化": 22,
            "汽车技术": 23,
            "用车养车": 24,
            "学车": 25,
            "汽车服务": 26,
            "花边": 27,
            "车展": 28,
            "人车生活": 29,
            "财经": 30,
            "汽车": 31
        }

    # Return tree is leave or not
    @staticmethod
    def _is_not_leave(tree):
        return type(tree[1]) == list

    def get_subtrees(self, tree):
        yield tree
        if self._is_not_leave(tree):
            for subtree in tree[1]:
                if self._is_not_leave(subtree):
                    for x in self.get_subtrees(subtree):
                        yield x

    # Returns pairs of paths and values of a tree
    def get_paths(self, tree):
        for i, subtree in enumerate(tree[1]):
            yield [i], subtree[0]
            if self._is_not_leave(subtree):
                for path, value in self.get_paths(subtree):
                    yield [i] + path, value

    # Returns the number of nodes in a tree (not including root)
    def count_nodes(self, tree):
        return self._count_nodes(tree[1])+1

    def _count_nodes(self, branches):
        if id(branches) in self._count_nodes_dict:
            return self._count_nodes_dict[id(branches)]
        size = 0
        for node in branches:
            if self._is_not_leave(node):
                size += 1 + self._count_nodes(node[1])
        self._count_nodes_dict[id(branches)] = size
        return size

    # Returns all the nodes in a path
    def get_nodes(self, tree, path):
        next_node = 0
        nodes = []
        for decision in path:
            nodes.append(next_node)
            if not self._is_not_leave(tree):
                break
            next_node += 1 + self._count_nodes(tree[1][:decision])
            tree = tree[1][decision]
        return nodes
