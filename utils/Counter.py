class Counter:
    def __init__(self):
        self.ln_cnt = 0
        self.foldable_cnt = 0
        self.center_modules = set()
        self.layer_info = []
        self.hooks = []
        self.indent = 0
