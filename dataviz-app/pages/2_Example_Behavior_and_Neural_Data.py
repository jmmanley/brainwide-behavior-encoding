import streamlit as st
import plots

st.markdown('# Example Behavior and Neural Data')

st.markdown('To get a sense of the type of data that are analyzed in systems neuroscience, below I show some example data from [Stringer & Pachitariu et al. 2019](https://www.science.org/doi/10.1126/science.aav7893), who monitored neural activity in populations of about 10,000 neurons in visual cortex of a mouse while it performed spontaneous and uninstructed behaviors. Stringer and Pachitariu\'s major insight was that a large fraction of neural variance within the visual cortex of the mouse (and, as it turns out, across the brain) can be explained just by the animal\'s own behavior. While it is expected that self-motion variables must be combined with sensory information somewhere in the brain in order to enable flexible behavior, it was surprising that this could be done as early as primary sensory cortex!')

st.markdown('As a proxy for the mouse\'s movements and arousal, the top row displays its Running speed (purple) and Pupil area (red). Below in the middle row are the latent variables identified via dimensionality reduction of the full neural data, called the Shared Variance Components (SVCs). Finally, the bottom row shows traces of actual spiking from individual example neurons.')

fig = plots.plot_example_behavior_neurons()
st.plotly_chart(fig)

st.markdown('Note the strong correlations between some of the Neural SVCs (e.g., SVCs 1 & 2) and the behavioral variables. Can you spot any single neuron in the bottom plot that is correlated with behavior? As you may notice, these individual neurons are much noisier and many of them are not clearly correlated with the behavior; however, dimensionality reduction of these noisy neurons reveals the latent representation of behavior within the SVCs.')

st.markdown('Some tips for interacting with these plots: click and drag horizontally within a plot to select a time period. Click and drag on the x or y axes to scroll in time or across neural variables, respectively.  To reset, click the home🏠 on the top right of the plot.')

st.markdown('### Source data')
st.markdown('Data retrieved from [Stringer & Pachitariu et al. 2018a on Figshare](https://janelia.figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_during_spontaneous_behaviors/6163622). This recording is `spont_M150824_MP019_2016-04-05.mat`. SVCs were calculated using the code in `run_svca_for_dataviz.py`. Go explore these data for yourself!')
