import visdom
import numpy as np

class Visualizer():
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = 1

    def plot_curves(self, Y, x, title='loss', xlabel='iterations', ylabel='accuracy'):
        keys = list(Y.keys())
        vals = list(Y.values())
        if len(vals) == 1:
            y = np.array(vals)
        else:
            y = np.array(vals).reshape(-1, len(vals))
        self.vis.line(Y=y,
                      X=np.array([self.index]),
                      win=title,
                      opts=dict(legend=keys, title = title, xlabel=xlabel, ylabel=ylabel),
                      update=None if self.index == 0 else 'append')
        self.index = x


if __name__ == '__main__':
    vis = Visualizer(env='test1')
    for i in range(10):
        y1 = i
        y2 = 2 * i
        y3 = 6 * i
        vis.plot_curves({'acc_lfw': y1, 'acc_agedb': y2}, x=i, title='train')
        vis.plot_curves({'acc_lfw': y1, 'acc_agedb': y2, 'acc_cfpfp': y3}, x=i, title='test')