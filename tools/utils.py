class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.max_val = 0
        self.min_val = 999999
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.max_val = max(self.max_val, val)
        self.min_val = min(self.min_val, val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count