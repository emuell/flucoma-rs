# Wrapped algorithms - Coverage Status

### Audio features (`flucoma_rs::analyzation`)

- [x] [`Loudness`](https://learn.flucoma.org/reference/loudness) as `flucoma_rs::analyzation::Loudness` -- EBU R128-style loudness + peak per frame
- [x] [`MelBands`](https://learn.flucoma.org/reference/melbands) as `flucoma_rs::analyzation::MelBands` -- mel-scaled filter bank (magnitude -> band energies)
- [x] [`OnsetDetectionFunctions`](https://learn.flucoma.org/reference/onsetfeature) as `flucoma_rs::analyzation::OnsetDetectionFunctions` -- 10 spectral-difference onset detection functions
- [x] [`STFT`](https://learn.flucoma.org/learn/fourier-transform/) as `flucoma_rs::analyzation::Stft` -- frame-by-frame Short-Time Fourier Transform
- [x] [`ISTFT`](https://learn.flucoma.org/learn/fourier-transform/) as `flucoma_rs::analyzation::Istft` -- inverse STFT, complex spectrum -> audio
- [ ] [`SpectralShape`](https://learn.flucoma.org/reference/spectralshape) -- 7 shape descriptors: centroid, spread, skewness, kurtosis, rolloff, flatness, crest
- [ ] [`ChromaFilterBank`](https://learn.flucoma.org/reference/chroma) -- chroma (pitch-class) filter bank
- [ ] [`DCT`](https://learn.flucoma.org/reference/dct) -- Discrete Cosine Transform
- [ ] [`YINFFT`](https://learn.flucoma.org/reference/pitch) -- YIN pitch estimator (spectral domain)
- [ ] [`CepstrumF0`](https://learn.flucoma.org/reference/pitch) -- cepstral fundamental frequency estimator
- [ ] [`HPS`](https://learn.flucoma.org/reference/pitch) -- Harmonic Product Spectrum pitch estimator
- [ ] [`SineFeature`](https://learn.flucoma.org/reference/sinefeature) -- sinusoidal peak feature extraction
- [ ] [`NoveltyFeature`](https://learn.flucoma.org/reference/noveltyfeature) -- self-similarity novelty feature

### Audio source separation (`flucoma_rs::decomposition`)

- [ ] [`HPSS`](https://learn.flucoma.org/reference/hpss) -- Harmonic-Percussive Source Separation
- [ ] [`SineExtraction`](https://learn.flucoma.org/reference/sines) -- sinusoids + residual decomposition
- [ ] [`TransientExtraction`](https://learn.flucoma.org/reference/transients) -- transient + residual decomposition
- [ ] [`NMF`](https://learn.flucoma.org/learn/bufnmf/) -- Non-negative Matrix Factorization
- [ ] [`NMFCross`](https://learn.flucoma.org/reference/bufnmfcross/) -- cross-synthesis via NMF
- [ ] [`NMFMorph`](https://learn.flucoma.org/reference/nmfmorph/) -- NMF-based spectral morphing

### Audio transformation (`flucoma_rs::transformation`)

- [x] [`AudioTransport`](https://learn.flucoma.org/reference/audiotransport) as `flucoma_rs::decomposition::AudioTransport` -- optimal-transport spectral morphing
- [ ] [`BufNMFCross`](https://learn.flucoma.org/reference/bufnmfcross/) -- resynthesis of targets using a source's spectral bases
- [ ] [`NMFFilter`](https://learn.flucoma.org/reference/nmffilter/) -- resynthesises a signal against a set of spectral templates
- [ ] [`NMFMorph`](https://learn.flucoma.org/reference/nmfmorph/) -- cross-synthesis using non-negative Matrix Factorisation

### Audio slicing (`flucoma_rs::segmentation`)

- [x] [`OnsetSegmentation`](https://learn.flucoma.org/reference/onsetslice) as `flucoma_rs::segmentation::OnsetSegmentation` -- segment audio stream at detected onsets
- [x] [`EnvelopeSegmentation`](https://learn.flucoma.org/reference/ampslice) as `flucoma_rs::segmentation::EnvelopeSegmentation` -- amplitude-envelope-based segmentation
- [x] [`NoveltySegmentation`](https://learn.flucoma.org/reference/noveltyslice) as `flucoma_rs::segmentation::NoveltySegmentation` -- novelty-curve segmentation
- [x] [`TransientSegmentation`](https://learn.flucoma.org/reference/transientslice) as `flucoma_rs::segmentation::TransientSegmentation` -- transient detector and segmenter

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
