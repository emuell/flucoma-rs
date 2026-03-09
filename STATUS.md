# Wrapped algorithms - Coverage Status

### Audio features (`flucoma_rs::analyzation`)

- [x] [`Loudness`](https://learn.flucoma.org/reference/loudness) as `flucoma_rs::analyzation::Loudness` -- EBU R128-style loudness + peak per frame
- [x] [`MelBands`](https://learn.flucoma.org/reference/melbands) as `flucoma_rs::analyzation::MelBands` -- mel-scaled filter bank (magnitude -> band energies)
- [x] [`Onset`](https://learn.flucoma.org/reference/onsetfeature) as `flucoma_rs::analyzation::Onset` -- 10 spectral-difference onset detection functions
- [x] [`NoveltyFeature`](https://learn.flucoma.org/reference/noveltyfeature) as `flucoma_rs::analyzation::Novelty` -- self-similarity novelty feature
- [x] [`SineFeature`](https://learn.flucoma.org/reference/sinefeature) as `flucoma_rs::analyzation::Sine` -- sinusoidal peak feature extraction
- [x] [`AmpFeature`](https://learn.flucoma.org/reference/ampfeature) as `flucoma_rs::analyzation::AmpFeature` -- amplitude envelope follower
- [ ] [`SpectralShape`](https://learn.flucoma.org/reference/spectralshape) -- 7 shape descriptors: centroid, spread, skewness, kurtosis, rolloff, flatness, crest
- [ ] [`ChromaFilterBank`](https://learn.flucoma.org/reference/chroma) -- chroma (pitch-class) filter bank
- [ ] [`YINFFT`](https://learn.flucoma.org/reference/pitch) -- YIN pitch estimator (spectral domain)
- [ ] [`CepstrumF0`](https://learn.flucoma.org/reference/pitch) -- cepstral fundamental frequency estimator
- [ ] [`HPS`](https://learn.flucoma.org/reference/pitch) -- Harmonic Product Spectrum pitch estimator

### Audio source separation (`flucoma_rs::decomposition`)

- [ ] [`HPSS`](https://learn.flucoma.org/reference/hpss) -- Harmonic-Percussive Source Separation
- [ ] [`SineExtraction`](https://learn.flucoma.org/reference/sines) -- sinusoids + residual decomposition
- [ ] [`TransientExtraction`](https://learn.flucoma.org/reference/transients) -- transient + residual decomposition
- [ ] [`NMF`](https://learn.flucoma.org/reference/bufnmf) -- Non-negative Matrix Factorization

### Audio transformation (`flucoma_rs::transformation`)

- [x] [`AudioTransport`](https://learn.flucoma.org/reference/audiotransport) as `flucoma_rs::transformation::AudioTransport` -- optimal-transport spectral morphing
- [x] [`NMFFilter`](https://learn.flucoma.org/reference/nmffilter) as `flucoma_rs::transformation::NMFFilter` -- resynthesises a signal against a set of spectral templates
- [x] [`NMFMorph`](https://learn.flucoma.org/reference/nmfmorph) as `flucoma_rs::transformation::NMFMorph` -- cross-synthesis using non-negative Matrix Factorisation
- [ ] [`BufNMFCross`](https://learn.flucoma.org/reference/bufnmfcross) -- resynthesis of targets using a source's spectral bases 

### Audio slicing (`flucoma_rs::segmentation`)

- [x] [`OnsetSlice`](https://learn.flucoma.org/reference/onsetslice) as `flucoma_rs::segmentation::OnsetSlice` -- segment audio stream at detected onsets
- [x] [`AmpSlice`](https://learn.flucoma.org/reference/ampslice) as `flucoma_rs::segmentation::AmpSlice` -- amplitude-envelope-based segmentation
- [x] [`NoveltySlice`](https://learn.flucoma.org/reference/noveltyslice) as `flucoma_rs::segmentation::NoveltySlice` -- novelty-curve segmentation
- [x] [`TransientSlice`](https://learn.flucoma.org/reference/transientslice) as `flucoma_rs::segmentation::TransientSlice` -- transient detector and segmenter

### Machine Learning & Statistics (`flucoma_rs::data`)

- [ ] [`MultiStats`](https://learn.flucoma.org/reference/stats) -- aggregate statistics (mean, std, min, max, ...) over a descriptor buffer
- [ ] [`RunningStats`](https://learn.flucoma.org/reference/stats) -- incremental running statistics
- [ ] [`Normalization`](https://learn.flucoma.org/reference/normalize) -- min-max feature normalization
- [ ] [`Standardization`](https://learn.flucoma.org/reference/standardize) -- Z-score standardization
- [ ] [`RobustScaling`](https://learn.flucoma.org/reference/robustscale) -- robust (median/IQR) feature scaling
- [ ] [`LabelSetEncoder`](https://learn.flucoma.org/reference/labelset) -- categorical label encoder
- [ ] [`KDTree`](https://learn.flucoma.org/reference/kdtree) -- K-D Tree for nearest-neighbour search
- [ ] [`KMeans`](https://learn.flucoma.org/reference/kmeans) -- K-Means clustering
- [ ] [`SKMeans`](https://learn.flucoma.org/reference/skmeans) -- Spherical K-Means clustering
- [ ] [`KNNClassifier`](https://learn.flucoma.org/reference/knnclassifier) -- K-Nearest Neighbour classifier
- [ ] [`KNNRegressor`](https://learn.flucoma.org/reference/knnregressor) -- K-Nearest Neighbour regressor
- [ ] [`MLP`](https://learn.flucoma.org/reference/mlpclassifier) -- Multi-Layer Perceptron
- [ ] [`SGD`](https://learn.flucoma.org/reference/mlpclassifier) -- Stochastic Gradient Descent optimiser (used by MLP)
- [ ] [`PCA`](https://learn.flucoma.org/reference/pca) -- Principal Component Analysis
- [ ] [`MDS`](https://learn.flucoma.org/reference/mds) -- Multidimensional Scaling
- [ ] [`UMAP`](https://learn.flucoma.org/reference/umap) -- UMAP dimensionality reduction

### Fourier Transform (`flucoma_rs::fourier`)

- [x] [`STFT`](https://learn.flucoma.org/learn/fourier-transform/) as `flucoma_rs::fourier::Stft` -- frame-by-frame Short-Time Fourier Transform
- [x] [`ISTFT`](https://learn.flucoma.org/learn/fourier-transform/) as `flucoma_rs::fourier::Istft` -- inverse STFT, complex spectrum -> audio
- [ ] [`GriffinLim`](https://learn.flucoma.org/learn/fourier-transform/) -- Griffin-Lim phase reconstruction (magnitude spectrum -> audio)
- [ ] [`DCT`](https://learn.flucoma.org/reference/mfcc) -- Discrete Cosine Transform (used by MFCC)
