import plotly.graph_objs as go
import pandas as pd

from hydrogram_biuld import HydrogramBiuld


class HydrogramClean(HydrogramBiuld):

    def __init__(self, data):
        if data is pd.Series:
            self.data = pd.DataFrame(data)
        else:
            self.data = data
        super().__init__()

    def plot(self, type_criterion=None):
        bandxaxis = go.layout.XAxis(title="Data")
        bandyaxis = go.layout.YAxis(title="Vazão(m³/s)")

        try:

            layout = dict(title="Hidrograma",
                          xaxis=bandxaxis, yaxis=bandyaxis,
                          font=dict(family='Time New Roman', color='rgb(0,0,0)')
                          )

            data = list()
            data.append(self._plot_one(self.data))
            fig = dict(data=data, layout=layout)
            return fig, data

        except AttributeError:
            name = 'Hidrograma'
            layout = dict(title=name,
                          xaxis=bandxaxis, yaxis=bandyaxis,
                          font=dict(family='Time New Roman'))

            data = list()
            data.append(self._plot_multi())
            fig = dict(data=data, layout=layout)
            return fig, data

    def _plot_multi(self):
        data = list()
        for i in self.data:
            data += self._plot_one(self.data[i])
        return data
