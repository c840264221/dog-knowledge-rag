class Trace:

    def __init__(self, trace_id):

        self.trace_id = trace_id

        self.root_spans = []

    def add_root_span(self, span):

        self.root_spans.append(span)

    def to_dict(self):

        return {

            "trace_id": self.trace_id,

            "spans": [

                span.to_dict()

                for span in self.root_spans
            ]
        }