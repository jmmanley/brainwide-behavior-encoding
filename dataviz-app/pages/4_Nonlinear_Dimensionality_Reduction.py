import streamlit as st
import plots

st.markdown('# Nonlinear Dimensionality Reduction')
st.markdown('## How low can we go?')

st.markdown('An open question in the field of neuroscience is how much lower can we reduce the dimensionality of our datasets utilizing nonlinear methods. Given the highly nonlinear and complex relationships within and among neurons, it is likely that circuit dynamics also exhibit highly nonlinear activity patterns. However, Stringer & Pachitariu et al. focused exclusively on linear representations of their data.')

st.markdown('Given that SVCA provides a useful quantitative approach to assess the reliability of each dimension, I have tested how much further we can reduce the dimensionality of the previously-identified neural SVCs while maintaining the same amount of reliable variance. To do so, I utilized a [variational autoencoder (VAE)](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73) approach, which maps the neural SVCs into another lower-dimensional nonlinear latent space. A VAE contains an encoder-decoder architecture. We can thus compare the reconstructed SVCs after being encoded and decoded through a VAE with any given latent dimensionality.')

st.markdown('Below, you can explore how the neural SVC reconstructions (in blue, with the real SVCs in white) change as a function of the dimensionality of the latent nonlinear mapping learned by the VAE.')

fig = plots.plot_reconstructed_svcs_vs_latent_dim()
st.plotly_chart(fig)


st.markdown('Finally, we can assess how many SVCs can be encoded by a given number of nonlinear VAE dimensions. Below, in purple we show the number of reliable SVCs as a function of the linear embedding dimensionality (essentially y=x until the maximum number of SVCs). In red, we show the number of reliable SVCs that can be encoded by a given nonlinear VAE embedding dimensionality. While the first ~100 SVCs appear highly compressible in a nonlinear space (e.g. 101 reliable SVCs can be encoded with 8 nonlinear dimensions), the following number of reliable SVCs then grows quite slowly with embedding dimensionality. Eventually, the VAE does not even reconstruct as many reliable SVCs as its latent dimensionality, suggesting it may be overfitting to the data. In this case, an SVC (or reconstructed SVC) is considered reliable if it contains at least 5% reliable variance on the held-out testing timepoints. Thus, while the lower SVCs (which, you may remember, are those that are most predictable from the mouse\'s behavior) may be highly compressible in a nonlinear space, the remaining SVCs do not appear to be explained by a much lower nonlinear dimensionality.')

fig = plots.plot_nreliable_vae_embedding()
st.plotly_chart(fig)

st.markdown('### Source data')
st.markdown('Data retrieved from [Stringer & Pachitariu et al. 2018a on Figshare](https://janelia.figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_during_spontaneous_behaviors/6163622). This recording is `spont_M161025_MP030_2017-06-16.mat`. SVCs and VAEs were calculated using the code in `vae.py`. Go explore these data for yourself!')
