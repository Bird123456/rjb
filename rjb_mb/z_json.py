import json

class total:
    def __init__(self, name, num):
        self.name = name
        self.num = num
    def __repr__(self):
        return repr((self.name, self.num))

# customers = [
#         total('xm', '21'),
#         total('xx', '22'),
#         ]
#
# json_str = json.dumps(customers, default=lambda o: o.__dict__, sort_keys=True, indent=4)
# print (json_str)
