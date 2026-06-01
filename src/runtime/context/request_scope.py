class RequestScope:

    def __init__(self):

        self._data = {}

    def set(self,key,value):

        self._data[key] = value

    def get(self,key,default=None):

        return self._data.get(key,default)

    def remove(self,key):

        self._data.pop(key,None)

    def clear(self):

        self._data.clear()