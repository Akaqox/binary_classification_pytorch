
class metrics():
    def __init__(self, results:tuple[int, int, int, int]) -> None:
        self.tp = results[0]
        self.tn = results[1]
        self.fp = results[2]
        self.fn = results[3]

    def accuracy(self) -> float:
        acc = (self.tp + self.tn)/(self.tp + self.tn + self.fp + self.fn)
        return acc
    
    def precision(self)-> float:
        prec = self.tp/(self.tp+self.fp)
        return prec

    def recall(self)-> float:
        rec = self.tp/(self.tp+self.fn)
        return rec
    
    def specifity(self)-> float:
        rec = self.tn/(self.tn+self.fp)
        return rec
    
    def F1(self)-> float:
        precision = self.precision()
        recall = self.recall()

        up = (2 * precision * recall)
        down = (precision + recall)
        f1 = up/down
        return f1
    
    def quick(self)-> None:
        acc = self.accuracy()
        print(f"Accuracy is = {acc:0,.3f}")

        prec = self.precision()
        print(f"Precision is = {prec:0,.3f}")

        rec = self.recall()
        print(f"Sensivity(Recall) is = {rec:0,.3f}")

        spec = self.specifity()
        print(f"Specifity is = {spec:0,.3f}")

        f1 = self.F1()
        print(f"F1 score is =  {f1 :0,.3f}")