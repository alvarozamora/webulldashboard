class slice():

    def __init__(self, ticker):

        self.ticker = ticker
        self.stock = 0
        self.collateral = 0
        self.options = 0

    def total(self, verbose=False):
        self.maint  = 0
        self.total  = self.stock + self.collateral + self.options
        self.maint += self.stock + self.collateral + self.options
        print(f"{self.ticker} total = ${self.total:.2f}; stock = ${self.stock:.2f}; collateral = ${self.collateral:.2f}; options = ${self.options:.2f}")

    def proportion(self, nw):
        self.stock      /= nw
        self.collateral /= nw
        self.options    /= nw
        self.total      /= nw
