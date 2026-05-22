class BaseMiddleware:

    def before(self, ctx):

        pass

    def after(self, ctx):

        pass

    def on_error(self, ctx, e):

        raise e