import streamlit as st
import os
import plots

st.markdown('# Introduction to Dimensionality Reduction in Neuroscience')

st.markdown('Due to impressive advances in the ability to monitor the activity of many neurons at cellular resolution, experimental neuroscientists can now regularly record many hundreds or thousands - and in [some cases](https://www.nature.com/articles/s41592-021-01239-8) up to 1 million(!) - neurons simultaneously! Optical microscopy provides a scalable and high spatiotemporal resolution approach to monitor the activity of neurons that have been induced to express a [Genetically Encodable Calcium Indicator (GECI)](https://en.wikipedia.org/wiki/Calcium_imaging#Genetically_encoded_calcium_indicators), such that the neuron fluoresces when its calcium levels are high, a useful proxy for neuronal spiking. See for example the video below from [Demas et al. 2021](https://www.nature.com/articles/s41592-021-01239-8); the blinking circles are spiking neurons in a 2x2mm region of mouse cortex, and the dark curves across the image are actually blood vessels (scale bar: 200mm, playback sped up 4x). This can be done in an awake, behaving animal as it performs tasks, which has provided huge insights into the neural dynamics underlying a wide range of adaptive behaviors.')

video = open(os.path.join(os.path.dirname(__file__),'demas2021_video9.mp4'), 'rb')
video_bytes = video.read()
st.video(video_bytes)

st.markdown('While many studies have focused on the properties of single neurons, it is now clear that the population-level neural dynamics can provide insight into the neural mechanisms underlying various behaviors. As such, dimensionality reduction has become a major tool in systems neuroscience in order to quickly visualize and quantify the "latent" (or hidden, in the sense that they are not directly measured) variables encoded across neurons.')

st.markdown('### Shared Variance Component Analysis')
st.markdown('[Stringer & Pachitariu et al. 2019](https://www.science.org/doi/10.1126/science.aav7893) developed Shared Variance Component Analysis (SVCA) as a neural dimensionality reduction technique to identify "dimensions of neural nariance that are reliably determined by common underlying signals." SVCA splits the neurons into two sets and identifies the dimensions of each set that maximally covary, called the Shared Variance Components (SVCs). Then, in order to determine whether each SVC contains true robust signal or just noise, the reliability of each SVC is quantified by the covariance between the projections of the two neural sets on held-out testing timepoints. This reliability can be visualized below by comparing the SVC projections of each cell set.')

fig = plots.plot_example_SVCs()
st.plotly_chart(fig, use_container_width=True)

st.markdown('The reliability of each SVC can be further quantified by calculating the percentage of variance in each SVC that is reliably covarying on the held-out testing timepoints. Below, on the left we see that more than 100 SVCs exhibit non-zero reliability, suggesting that this region of visual cortex encodes a high-dimensional signal. However, the total variance in each SVC decays rapidly, as shown in the absolute variance in each SVC on the right, suggesting that many of these reliable dimensions are very small signals.')

fig = plots.plot_reliable_variance()
st.plotly_chart(fig, use_container_width=True)

st.markdown('### Source data')
st.markdown('Data retrieved from [Stringer & Pachitariu et al. 2018a on Figshare](https://janelia.figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_during_spontaneous_behaviors/6163622). This recording is `spont_M150824_MP019_2016-04-05.mat`. SVCs were calculated using the code in `run_svca_for_dataviz.py`. Go explore these data for yourself!')

