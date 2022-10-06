import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from plotly.express.colors import sample_colorscale
from scipy.stats import zscore
import os

st.markdown('# Nonlinear Dimensionality Reduction')
st.markdown('## How low can we go?')

st.markdown('An open question in the field of neuroscience is how much lower can we reduce the dimensionality of our datasets utilizing nonlinear methods. Given the highly nonlinear and complex relationships within and among neurons, it is likely that circuit dynamics also exhibit highly nonlinear activity patterns. However, Stringer & Pachitariu et al. focused exclusively on linear representations of their data.')

st.markdown('Given that SVCA provides a useful quantitative approach to assess the reliability of each dimension, I have tested how much further we can reduce the dimensionality of the previously-identified neural SVCs while maintaining the same amount of reliable variance. To do so, I utilized a variational autoencoder (VAE) approach, which maps the neural SVCs into another lower-dimensional nonlinear latent space. A VAE contains an encoder-decoder architecture, such that we can then compare the reconstructed SVCs after being encoded and decoded through a VAE with any given latent dimensionality.')

st.markdown('Below, you can explore how the neural SVC reconstructions (in blue, with the real SVCs in white) change as a function of the dimensionality of the latent nonlinear mapping learned by the VAE.')

file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'processed-data', 'spont_M161025_MP030_2017-06-16_128SVC_VAE.npz')
data = np.load(file)

projs1 = zscore(data['projs1'],axis=1)
projs2_preds = zscore(data['projs2_preds'],axis=2)
t = np.arange(projs1.shape[1])/3

fig = make_subplots(rows=1, cols=1)

toplot = [0,1,15,31,63,127]
colors = sample_colorscale('Twilight',[(2*x%10)/10 for x in np.arange(len(toplot))])

for i in range(len(toplot)):

    fig.append_trace(go.Scatter(x=t,
        y=projs1[toplot[i],:]/7 - i, mode='lines',
        line = dict(color=colors[0]), name = "SVC "+str(toplot[i]+1)
    ), row=1, col=1)

    fig.append_trace(go.Scatter(x=t,
        y=projs2_preds[0,toplot[i],:]/7 - i, mode='lines',
        line = dict(color=colors[1]), name = "Predicted SVC "+str(toplot[i]+1)
    ), row=1, col=1)

# Define frames
number_frames = projs2_preds.shape[0]
frames = []

for k in range(number_frames):
    
    data = []
    
    for i in range(len(toplot)):
        data.append(go.Scatter(visible=True))
        data.append(go.Scatter(y=projs2_preds[k,toplot[i],:]/7 - i))

    frames.append(dict(
        name=k,
        data=data
    ))


# Play button
updatemenus = [dict(type='buttons',
                    buttons=[dict(label='Play',
                                  method='animate',
                                  args=[[f'{k}' for k in range(number_frames)],
                                        dict(frame=dict(duration=500, redraw=False),
                                             transition=dict(duration=0),
                                             easing='linear',
                                             fromcurrent=True,
                                             mode='immediate'
                                             )]),
                             dict(label='Pause',
                                  method='animate',
                                  args=[[None],
                                        dict(frame=dict(duration=0, redraw=False),
                                             transition=dict(duration=0),
                                             mode='immediate'
                                             )])
                             ],
                    direction='left',
                    pad=dict(r=10, t=85),
                    showactive=True, x=0.1, y=0, xanchor='right', yanchor='top')
               ]


# Slider
sliders = [{'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {'font': {'size': 16}, 'prefix': 'VAE latent dimensionality: ', 'visible': True, 'xanchor': 'right'},
            'transition': {'duration': 0, 'easing': 'linear'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9, 'x': 0.1, 'y': 0,
            'steps': [{'args': [[k], {'frame': {'duration': 0, 'easing': 'linear', 'redraw': True},
                                      'transition': {'duration': 0, 'easing': 'linear'}}],
                       'label': 2**k, 'method': 'animate'} for k in range(number_frames)
                      ]}]


fig.update(frames=frames)
fig.update_layout(height=800, width=1000, xaxis_title='Time (seconds)', yaxis_title='SVC dimension', 
                  showlegend=False, yaxis_showgrid=False, yaxis_zeroline=False, yaxis_tickvals=[-x for x in range(len(toplot))],
                  yaxis_ticktext=[x+1 for x in toplot], updatemenus=updatemenus,
                  sliders=sliders)

st.plotly_chart(fig)


st.markdown('Finally, we can assess how many SVCs can be encoded by a given number of nonlinear VAE dimensions. Below, in purple we show the number of reliable SVCs as a function of the linear embedding dimensionality (essentially y=x until the maximum number of SVCs). In red, we show the number of reliable SVCs that can be encoded by a given nonlinear VAE embedding dimensionality. While the first ~100 SVCs appear highly compressible in a nonlinear space (e.g. 101 reliable SVCs can be encoded with 8 nonlinear dimensions), the number of reliable SVCs then grows quite slowly with embedding dimensionality. Eventually, the VAE does not even reconstruct as many reliable SVCs as its latent dimensionality, suggesting it may be overfitting to the data. In this case, an SVC (or reconstructed SVC) is considered reliable if it contains at least 5% reliable variance on the held-out testing timepoints. Thus, while the lower SVCs (which, you may remember, are those that are predictable from the mouse\'s behavior) may be highly compressible in a nonlinear space, the remaining SVCs do not appear to be explained by a much lower nonlinear dimensionality.')

file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'processed-data', 'spont_M161025_MP030_2017-06-16_128SVC_VAE.npz')
data = np.load(file)
sneur = data['sneur'][:2048]
varneur = data['varneur'][:2048]

sneur_vae = data['sneur_vae'][:2048]
varneur_vae = data['varneur_vae'][:2048]

fig = make_subplots(rows=1, cols=1)

fig.append_trace(go.Scatter(x=np.arange(len(sneur))+1,
    y=np.cumsum(sneur/varneur>0.05), mode='lines',
    name = "Linear (SVCA)"
), row=1, col=1)

output_dims = 2**np.arange(12)

fig.append_trace(go.Scatter(x=output_dims,
    y=np.sum(sneur_vae / varneur_vae > 0.05,axis=1), mode='lines',
    name = "Nonlinear (VAE)"
), row=1, col=1)

fig.update_layout(width=600, xaxis_title='Embedding dimensionality', yaxis_title='Number of encoded SVCs (>5% reliable variance)', 
                  xaxis_type='log', yaxis_type='log')

st.plotly_chart(fig)

st.markdown('### Source data')
st.markdown('Data retrieved from [Stringer & Pachitariu et al. 2018a on Figshare](https://janelia.figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_during_spontaneous_behaviors/6163622). This recording is `spont_M161025_MP030_2017-06-16.mat`. SVCs and VAEs were calculated using the code in `vae.py`. Go explore these data for yourself!')
