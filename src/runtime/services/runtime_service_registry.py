class RuntimeServiceRegistry:

    def __init__(self):

        self._services = {}

    def register(self,service):

        self._services[type(service)] = service

    def get(self,service_type):

        return self._services.get(service_type)

    def all_services(self):

        return self._services.values()