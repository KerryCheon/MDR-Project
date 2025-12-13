# Jakob Balkovec
# Dec 13th 2025
# fourier_transform.py
#
# Fourier-based smoothing utilities
#
# Provides a small API compatible with the Whittaker smoother in this
# folder. The main function fourier_smooth performs low-pass filtering
# in the Fourier domain with several filter shapes and handles missing
# values by linear interpolation prior to transform.
#
# CURRENTLY STILL IN TESTING & DEVELOPMENT
# raise NotImplementedError("do not import this module...still in testing & development")

import numpy as np


def _interp_missing(x, mask):
	x = np.asarray(x, dtype=float)
	mask = np.asarray(mask, dtype=bool)
	n = x.shape[0]
	if mask.sum() == 0:
		raise ValueError("no observed values to interpolate from")
	if mask.all():
		return x.copy()

	xi = np.arange(n)
	xp = xi[mask]
	fp = x[mask]
	# np.interp uses linear interpolation and fills out of bounds with first/last
	interp = np.interp(xi, xp, fp)
	return interp


def _make_filter(freqs, cutoff, kind="gaussian", order=2):
	if cutoff is None:
		# No filtering
		return np.ones_like(freqs)

	cutoff = float(cutoff)
	if kind == "ideal":
		return (freqs <= cutoff).astype(float)
	elif kind == "gaussian":
		# Gaussian low-pass: exp(-0.5*(f/sig)^2) where cutoff ~ 2*sig
		sigma = max(1e-12, cutoff / 2.0)
		return np.exp(-0.5 * (freqs / sigma) ** 2)
	elif kind == "butterworth":
		order = max(1, int(order))
		return 1.0 / (1.0 + (freqs / max(1e-12, cutoff)) ** (2 * order))
	else:
		raise ValueError("unknown filter kind: %s" % (kind,))


def fourier_smooth(x, cutoff=0.1, mask=None, kind="gaussian", order=2, pad=True):
	# params:
	# - x: array-like, 1D numeric. May contain np.nan where data is missing.
	# - cutoff: normalized cutoff frequency in cycles/sample (0..0.5). None
	# 	disables filtering (passes signal through). Default 0.1.
	# - mask: optional 1D boolean array or 0/1 weights same length as x.
	# 	If None, inferred as ~np.isnan(x).
	# - kind: string, one of ("gaussian", "ideal", "butterworth").
	# - order: integer order for Butterworth filter (ignored for other kinds).
	# - pad: bool, whether to pad the series (by reflection) to reduce wrap-around
	# 	effects from the FFT. Default True.

	# returns:
	# - z: numpy array of the same shape as x containing the smoothed series.

	# notes:
	# - This implementation treats missing values by first linearly
	#   interpolating the input according to mask and smoothing the result.
	#   This is a pragmatic choice when a proper weighted Fourier solution
	#   is not available. Values originally missing still get a smoothed
	#   estimate in the returned result.
	# - The result is real-valued and returned as float ndarray.

	x = np.asarray(x, dtype=float)
	n = x.shape[0]
	if n == 0:
		return np.asarray(x)

	if mask is None:
		mask = ~np.isnan(x)
	mask = np.asarray(mask, dtype=bool)

	# Interpolate missing values for FFT step
	y = _interp_missing(x, mask)

	# Detrend by subtracting mean; this reduces low-frequency bias
	mean_y = np.mean(y)
	y0 = y - mean_y

	# Optionally pad to reduce circular convolution artifacts
	if pad:
		# reflect padding: create mirrored extension at both ends
		pad_len = min(2 ** int(np.ceil(np.log2(n))) - n, n)
		if pad_len > 0:
			left = y0[:pad_len][::-1]
			right = y0[-pad_len:][::-1]
			y_pad = np.concatenate([left, y0, right])
		else:
			y_pad = y0
	else:
		y_pad = y0

	# compute real FFT and frequency grid (normalized to Nyquist)
	N = y_pad.shape[0]
	freqs = np.fft.rfftfreq(N, d=1.0)  # cycles/sample (0..0.5)
	filt = _make_filter(freqs, cutoff, kind=kind, order=order)

	Y = np.fft.rfft(y_pad)
	Zf = Y * filt
	z_pad = np.fft.irfft(Zf, n=N)

	if pad and (pad_len > 0):
		z = z_pad[pad_len:pad_len + n]
	else:
		z = z_pad[:n]

	# add mean back and return
	z = z + mean_y
	return z


def fourier_smooth_like_whittaker(x, lmbda=10, mask=None):
	# heuristic: map lmbda ( >0) to cutoff in (0.5, tiny) via 1/(1+alpha*sqrt(lmbda))
	if lmbda is None:
		cutoff = 0.1
	else:
		# scale factor chosen to map typical lmbda values around 10..1000
		cutoff = 0.5 / (1.0 + 0.05 * np.sqrt(max(1e-12, lmbda)))
		cutoff = float(max(1e-6, min(0.5, cutoff)))
	return fourier_smooth(x, cutoff=cutoff, mask=mask, kind="gaussian")


if __name__ == "__main__":
	# Minimal smoke test / example when run as a script
	import matplotlib.pyplot as plt

	t = np.linspace(0, 1, 200)
	signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 20 * t)
	# add trend
	signal = signal + 0.5 * t
	# add missing values
	signal[[10, 23, 45, 80]] = np.nan

	z = fourier_smooth_like_whittaker(signal, lmbda=50)

	plt.plot(t, signal, label="input")
	plt.plot(t, z, label="fourier smooth")
	plt.legend()
	plt.show()

